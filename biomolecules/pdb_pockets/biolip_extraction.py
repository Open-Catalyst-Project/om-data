import pandas as pd
import re
import os
import random
import subprocess
import glob
import shutil
import sys
from schrodinger.structure import StructureReader, StructureWriter
from schrodinger.structutils import build, analyze
from schrodinger.protein.captermini import CapTermini
from schrodinger.application import prepwizard
from schrodinger.adapter import evaluate_smarts

SCHRO = "/opt/schrodinger2024-1"


def get_biolip_db(lig_type="reg"):
    columns = [
        "pdb_id",
        "receptor_chain",
        "resolution",
        "binding_site_number_code",
        "ligand_id",
        "ligand_chain",
        "ligand_serial_number",
        "binding_site_residues_pdb",
        "binding_site_residues_re-numbered",
        "catalytic_site_residues_pdb",
        "catalytic_site_residues_re-numbered",
        "ec_number",
        "go_terms",
        "binding_affinity_manual",
        "binding_affinity_binding_moad",
        "binding_affinity_pdbbind_cn",
        "binding_affinity_bindingdb",
        "uniprot_ids",
        "pubmed_id",
        "ligand_residue_number",
        "receptor_sequence",
    ]
    types = {col: str for col in columns}
    types["ligand_serial_number"] = int
    biolip_df = pd.read_csv(
        "/home/levineds/BioLiP.txt",
        sep="\t",
        names=columns,
        dtype=types,
        keep_default_na=False,
    )
    biolip_df[["ligand_residue_number", "ligand_residue_end"]] = biolip_df[
        "ligand_residue_number"
    ].str.split("~", expand=True)
    biolip_df["ligand_residue_end"] = biolip_df["ligand_residue_end"].fillna(
        biolip_df["ligand_residue_number"]
    )
    biolip_df[["ligand_residue_number", "ligand_residue_inscode"]] = biolip_df[
        "ligand_residue_number"
    ].str.extract(r"(-?\d+)([A-Z\s]+)$")
    biolip_df["ligand_residue_end"] = biolip_df["ligand_residue_end"].str.extract(
        r"(-?\d+)"
    )
    biolip_df[["ligand_residue_number", "ligand_residue_end"]] = biolip_df[
        ["ligand_residue_number", "ligand_residue_end"]
    ].astype(int)
    # biolip_df.drop_duplicates(subset=['ligand_id', 'binding_site_residues_pdb'], inplace=True)
    biolip_df["has_binding_info"] = (
        biolip_df[
            [
                "binding_affinity_manual",
                "binding_affinity_binding_moad",
                "binding_affinity_pdbbind_cn",
                "binding_affinity_bindingdb",
            ]
        ]
        != ""
    ).any(axis=1)

    if lig_type == "macromol":
        biolip_df = biolip_df[biolip_df["ligand_id"].isin(["dna", "rna", "peptide"])]
    elif lig_type == "reg":
        biolip_df = biolip_df[~biolip_df["ligand_id"].isin(["dna", "rna", "peptide"])]
    return biolip_df


def retreive_ligand_and_env(biolip_df, ligand_size_limit=50, start_pdb=0, end_pdb=1000):
    random.seed(12341)
    grouped_biolip = biolip_df.groupby("pdb_id")
    for pdb_count in range(start_pdb, end_pdb):
        # pdb_id = random.choice(list(grouped_biolip.groups.keys()))
        pdb_id = list(grouped_biolip.groups.keys())[pdb_count]
        rows = grouped_biolip.get_group(pdb_id)
        print(f"preparing {pdb_id} (entry: {pdb_count})")
        sys.stdout.flush()
        prepped_pdb_fnames = set()
        for idx, row in rows.iterrows():
            try:
                st, fname = get_prepped_protein(pdb_id, row)
            except ValueError as e:
                print(f"Error on {pdb_id}: Structure has only CA")
                print(e)
                continue
            except RuntimeError as e:
                print(f"Error on {pdb_id}: PrepWizard failed")
                print(e)
                continue
            else:
                prepped_pdb_fnames.add(fname)
            binding_site_counter = (
                row["binding_site_number_code"] + row["receptor_chain"]
            )
            try:
                lig_ats, res_ats, st = get_atom_lists(st, row)
            except RuntimeError as e:
                print(f"Error on {pdb_id}, BS{binding_site_counter}: Atoms are missing")
                print(e)
                continue
            heavy_lig_ats = [at for at in lig_ats if st.atom[at].atomic_number > 1]
            if len(heavy_lig_ats) > ligand_size_limit:
                print("ligand too big: ", len(heavy_lig_ats))
                continue
            print(f"obtaining {pdb_id}, binding site {binding_site_counter}")
            ligand_env = st.extract(lig_ats + res_ats)
            try:
                cap_termini(st, ligand_env)
            except Exception as e:
                print(
                    f"Error on {pdb_id}, BS{binding_site_counter}: Cannot cap termini"
                )
                print(e)
                continue
            ligand_env = build.reorder_protein_atoms_by_sequence(ligand_env)
            fname = f"{pdb_id}_{binding_site_counter}_{ligand_env.formal_charge}.pdb"
            ligand_env.write(fname)

        # Cleanup the remaining prepped files for this pdb_id
        for fname in prepped_pdb_fnames:
            os.remove(fname)


def get_prepped_protein(pdb_id, row, prep=True):
    baseoutname = f"{pdb_id}_prepped.maegz"
    pdb_name = pdb_id
    chains = {row["receptor_chain"], row["ligand_chain"]}
    if len(chains) == 1:
        solo_chain = next(iter(chains))
        pdb_name += f":{solo_chain}"
        outname = f"{pdb_id}_{solo_chain}_prepped.maegz"
    else:
        outname = baseoutname
    basename = pdb_name.replace(":", "_")
    fname = basename + ".pdb"

    if not prep:
        subprocess.run([os.path.join(SCHRO, "utilities", "getpdb"), pdb_id])
        outname = f"{pdb_id}.pdb"
    elif os.path.exists(outname):
        # We've already done this example
        pass
    elif os.path.exists(baseoutname):
        # We've not done this chain, but we have done the whole protein
        outname = baseoutname
    else:
        # We haven't done anything to prepare the protein
        subprocess.run([os.path.join(SCHRO, "utilities", "getpdb"), pdb_name])
        if not os.path.exists(fname):
            # PDB could not be downloaded, try the cif
            subprocess.run(
                [os.path.join(SCHRO, "utilities", "getpdb"), pdb_id, "-format", "cif"]
            )
            st = StructureReader.read(f"{pdb_id}.cif")
            # only extract the relevant chains because these proteins tend to be very large and PrepWizard takes hours
            at_list = []
            for chain in chains:
                at_list.extend(st.chain[chain].getAtomList())
            st = st.extract(at_list)
            fname = f"{pdb_id}.maegz"
            with StructureWriter(fname) as writer:
                writer.append(st)
        st = StructureReader.read(fname)
        # Reject structure that are just a bunch of CA
        if (
            sum(
                1
                for mol in st.molecule
                if len(mol.atom) == 1 and mol.atom[1].element == "C"
            )
            > 20
        ):
            raise ValueError
        dummy_atoms = [at for at in st.atom if at.atomic_number < 1]
        st.deleteAtoms(dummy_atoms)
        st.write(fname)
        subprocess.run(
            [
                os.path.join(SCHRO, "utilities", "prepwizard"),
                fname,
                outname,
                "-noepik",
                "-noimpref",
                "-fillsidechains",
                "-disulfides",
                "-NOJOBID",
            ]
        )
    if not os.path.exists(outname):
        raise RuntimeError

    st = StructureReader.read(outname)
    deprotonate_phosphate_esters(st)
    st = build.reorder_protein_atoms_by_sequence(st)

    # Cleanup
    for file_to_del in (fname, f"{pdb_id}.pdb", f"{pdb_id}.cif"):
        if os.path.exists(file_to_del):
            os.remove(file_to_del)
    for deldir in glob.glob(f"{basename}-???"):
        shutil.rmtree(deldir)
    return st, outname


def deprotonate_phosphate_esters(st):
    """
    At physiological pH, it's a good assumption that any phosphate esters
    will be deprotonated. In the absence pKa's for ligands, we will make
    this assumption.
    """
    phos_smarts = "[*;!#1][*][P](=[O])([O])([O][H])"
    matched_ats = evaluate_smarts(st, phos_smarts)
    H_ats = {ats[-1] for ats in matched_ats}
    O_ats = {ats[-2] for ats in matched_ats}
    for O_at in O_ats:
        st.atom[O_at].formal_charge -= 1
    st.deleteAtoms(H_ats)


def get_atom_lists(st, row, include_waters=False):
    """
    Extract the ligand as its own chain.

    We extract a chain as DNA, RNA, peptides will have multiple residues.
    We also include an option to extract waters in case we want the waters
    associated with the ligand chain as well.
    """
    res_ats = []
    res_list = [res[1:] for res in row["binding_site_residues_pdb"].split()]
    gap_res = get_single_gaps(st, row["receptor_chain"], res_list)
    for res_num in gap_res:
        res = st.findResidue(f'{row["receptor_chain"]}:{res_num}')
        st_copy = st.copy()
        if not res.isStandardResidue():
            ## You have to be pretty weird to not be a Standard Residue
            continue
        try:
            build.mutate(st, res.getAlphaCarbon(), "GLY")
        except:
            # If the residue being turned to GLY is not just a simple amino
            # acid, e.g. it is bound to a ligand, it can't be mutated properly
            # and it will actually also garble the structure
            st = st_copy
            res = st.findResidue(f'{row["receptor_chain"]}:{res_num}')
            if res.getBetaCarbon() is not None:
                st.deleteBond(res.getBetaCarbon(), res.getAlphaCarbon())
                build.mutate(st, res.getAlphaCarbon(), "GLY")

    # Mutating residues can change the atom numbering so out of an abundance
    # of caution, let's retreive all the atom indices in a separate loop
    for res_num in gap_res + res_list:
        res = st.findResidue(f'{row["receptor_chain"]}:{res_num}')
        if res.hasMissingAtoms():
            print(f'Missing atoms in rec res {row["receptor_chain"]}:{res_num}')
            raise RuntimeError
        res_ats.extend(res.getAtomIndices())

    lig_ats = []
    for res_num in range(row["ligand_residue_number"], row["ligand_residue_end"] + 1):
        res_name = f'{row["ligand_chain"]}:{res_num}{row["ligand_residue_inscode"]}'
        try:
            res = st.findResidue(res_name)
        except ValueError:
            print(f"missing expected residue: {res_name}")
            continue
        lig_ats.extend(res.getAtomIndices())
    # Add in coodinating groups for ions
    if len(lig_ats) == 1:
        asl_str = f"fillres within 3 atom.num {lig_ats[0]}"
        coord_ats = analyze.evaluate_asl(st, asl_str)
        lig_ats.extend([at for at in coord_ats if at not in res_ats])

    # mark the ligand chain
    for at in lig_ats:
        st.atom[at].chain = "ligand"
    # mark the receptor chain
    for at in res_ats:
        st.atom[at].chain = "A"

    return lig_ats, res_ats, st


def get_single_gaps(st, rec_chain, res_list):
    parent_indices = set()
    parent_list = list(st.chain[rec_chain].residue)
    for res_name in res_list:
        res = st.findResidue(f"{rec_chain}:{res_name}")
        parent_indices.add(parent_list.index(res))
    gaps = [
        val + 1
        for val in parent_indices
        if val + 1 not in parent_indices and val + 2 in parent_indices
    ]
    gap_res = [f"{parent_list[gap].resnum}{parent_list[gap].inscode}" for gap in gaps]
    return gap_res


def cap_termini(st, ligand_env):
    capterm = CapTermini(ligand_env, verbose=False, frag_min_atoms=0)
    for res in capterm.cappedResidues():
        orig_res = st.findResidue(res)
        new_res = ligand_env.findResidue(res)
        try:
            val = st.measure(*orig_res.getDihedralAtoms("Psi"))
            ligand_env.adjust(val, *new_res.getDihedralAtoms("Psi"))

            val = st.measure(*reversed(orig_res.getDihedralAtoms("Phi")))
            ligand_env.adjust(val, *reversed(new_res.getDihedralAtoms("Phi")))
        except:
            pass


biolip_df = get_biolip_db()
batch_size = 1000
start_pdb = int(sys.argv[1])
end_pdb = int(sys.argv[2])
ligand_env = retreive_ligand_and_env(biolip_df, start_pdb=start_pdb, end_pdb=end_pdb)
