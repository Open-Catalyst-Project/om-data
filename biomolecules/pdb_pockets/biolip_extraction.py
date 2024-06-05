import glob
import os
import random
import re
import shutil
import subprocess
import sys

import pandas as pd
from schrodinger.adapter import evaluate_smarts
from schrodinger.application import prepwizard
from schrodinger.protein.captermini import CapTermini
from schrodinger.structure import Structure, StructureReader, StructureWriter
from schrodinger.structutils import analyze, build

SCHRO = "/opt/schrodinger2024-1"


class OnlyCAError(Exception):
    pass


class MissingAtomsError(Exception):
    pass


class MissingResiduesError(Exception):
    pass


def get_biolip_db(lig_type: str = "reg") -> pd.DataFrame:
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
        "BioLiP_05_31_2024.txt",
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
    biolip_df.drop_duplicates(
        subset=["ligand_id", "binding_site_residues_pdb"], inplace=True
    )
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


def retreive_ligand_and_env(
    biolip_df: pd.DataFrame,
    ligand_size_limit: int = 50,
    start_pdb: int = 0,
    end_pdb: int = 1000,
) -> None:
    """
    Extract ligand and its environment from BioLiP examples

    Procedure is as follows, for each protein in BioLiP:
        1) Download the structure and prepare it with PrepWizard (re-using already prepared proteins if possible)
        2) Obtain list of atoms which are assigned as ligands or receptors in BioLiP
        3) Extract the receptor residues and cap them to match the source protein
        4) Write the extracted structure to a pdb file
    """
    # random.seed(12341)
    grouped_biolip = biolip_df.groupby("pdb_id")
    pdb_list = list(grouped_biolip.groups.keys())
    for pdb_count in range(start_pdb, end_pdb):
        # pdb_id = random.choice(list(grouped_biolip.groups.keys()))
        pdb_id = pdb_list[pdb_count]
        rows = grouped_biolip.get_group(pdb_id)
        print(f"preparing {pdb_id} (entry: {pdb_count})")
        sys.stdout.flush()
        prepped_pdb_fnames = set()
        for idx, row in rows.iterrows():
            binding_site_counter = (
                row["binding_site_number_code"] + row["receptor_chain"]
            )
            try:
                st, fname = get_prepped_protein(pdb_id, row)
            except OnlyCAError:
                print(f"Error on {pdb_id}: Structure has only CA")
                continue
            except RuntimeError as e:
                print(f"Error on {pdb_id}: PrepWizard failed")
                print(e)
                continue
            else:
                prepped_pdb_fnames.add(fname)

            try:
                lig_ats, res_ats = get_atom_lists(st, row)
            except MissingAtomsError:
                print(f"Error on {pdb_id}, BS{binding_site_counter}: Atoms are missing")
                continue

            # Count heavy atoms of ligand and skip if too large
            heavy_lig_ats = [at for at in lig_ats if st.atom[at].atomic_number > 1]
            if len(heavy_lig_ats) > ligand_size_limit:
                print("ligand too big: ", len(heavy_lig_ats))
                continue

            # Do the extraction
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


def download_cif(pdb_id: str, chains: list[str]) -> str:
    """
    Download cif (as opposed to pdb) of a given protein and extract
    the needed chains.

    :param pdb_id: name of PDB to download
    :param chains: list of relevant chain names to extract
    :return: filename where structure has been downloaded
    """
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
    return fname


def missing_backbone_check(st: Structure):
    """
    Check if the protein backbone is present in the structure

    Rarely, some protein structures are just C alpha positions. We check
    for this by making sure that the structure doesn't contain lots of
    isolated carbon atoms.

    :param st: structure to check
    :raises: OnlyCAError if structure seems like it is only CA
    """
    if (
        sum(
            1
            for mol in st.molecule
            if len(mol.atom) == 1 and mol.atom[1].element == "C"
        )
        > 20
    ):
        raise OnlyCAError


def get_prepped_protein(
    pdb_id: str, row: pd.Series, prep: bool = True
) -> tuple[Structure, str]:
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
            fname = download_cif(pdb_id, chains)
        st = StructureReader.read(fname)
        # Reject structure that are just a bunch of CA
        missing_backbone_check(st)

        # Remove any dummy atoms, PrepWizard doesn't like them
        dummy_atoms = [at for at in st.atom if at.atomic_number < 1]
        st.deleteAtoms(dummy_atoms)
        st.write(fname)

        # Run PrepWizard
        try:
            run_prepwizard(fname, outname)
        except subprocess.TimeoutExpired:
            raise RuntimeError("PrepWizard took longer than 2 hours, skipping")
    if not os.path.exists(outname):
        raise RuntimeError("PrepWizard failed")

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


def run_prepwizard(fname: str, outname: str) -> None:
    """
    Run Schrodinger's PrepWizard

    In most cases, this seems to run in a handful of seconds to a few minutes.
    In a few extreme examples, it took hours or days. Since this holds up processing
    for the entire batch, we will add a 2 hour time-limit with PROPKA and an additional
    2 hour time-limit not using PROPKA (which seems to frequently be were things get bogged
    down, though it may also be in H-Bond optimization).

    :param fname: input file name for PrepWizard
    :param outname: output file name from PrepWizard
    """
    try:
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
            ],
            timeout=3600 * 2,  # Wait not more than 2 hours
        )
    except subprocess.TimeoutExpired as e:
        # Try again without PROPKA (which seems to be where things get stuck)
        subprocess.run(
            [
                os.path.join(SCHRO, "utilities", "prepwizard"),
                fname,
                outname,
                "-noepik",
                "-noimpref",
                "-fillsidechains",
                "-disulfides",
                "-nopropka",
                "-NOJOBID",
            ],
            timeout=3600 * 2,  # Wait not more than 2 hours
        )


def deprotonate_phosphate_esters(st: Structure) -> None:
    """
    At physiological pH, it's a good assumption that any phosphate esters
    will be deprotonated. In the absence pKa's for ligands, we will make
    this assumption.

    :param st: Structure with phosphate groups that can be deprotonated
    """
    phos_smarts = "[*;!#1][*][P](=[O])([O])([O][H])"
    matched_ats = evaluate_smarts(st, phos_smarts)
    H_ats = {ats[-1] for ats in matched_ats}
    O_ats = {ats[-2] for ats in matched_ats}
    for O_at in O_ats:
        st.atom[O_at].formal_charge -= 1
    st.deleteAtoms(H_ats)


def make_gaps_gly(st: Structure, row: pd.Series, gap_res: list[str]) -> None:
    """
    Turn gap residues into glycines

    If we have a single residue which would not normally be included
    which is between two which are, then the capping groups of those
    two are going to overlap because they are both trying to fill in
    the same missing CA. As such, we should just include that missing
    single gap residue as a glycine (it's actually smaller than two
    capping groups).

    If that gap residue is something non-standard (and Schrodinger
    considers a lot more than the 20 standard amino acids to be
    "standard", e.g. selenomethionine, hydroxyproline are "standard"),
    we just leave it as is.

    :param st: Structure being modified
    :param row: Series containing receptor and ligand information
    :param gap_res: list of residue numbers which are single gaps
    """
    for res_num in gap_res:
        res = st.findResidue(f'{row["receptor_chain"]}:{res_num}')
        ## You have to be pretty weird to not be a Standard Residue,
        ## just keep these as is
        if not res.isStandardResidue():
            continue
        # If the residue being turned to GLY is not just a simple amino
        # acid, e.g. it is bound to a ligand, it can't be mutated properly
        # and it will actually also garble the structure
        if res.getBetaCarbon() is not None:
            st.deleteBond(res.getBetaCarbon(), res.getAlphaCarbon())
        build.mutate(st, res.getAlphaCarbon(), "GLY")


def get_atom_lists(st: Structure, row: pd.Series) -> tuple[list[int], list[int]]:
    """
    Get the lists of ligand atoms and receptor atoms

    :param st: Structure to extract from
    :param row: Series containing receptor and ligand information
    """
    res_ats = []
    res_list = [res[1:] for res in row["binding_site_residues_pdb"].split()]
    gap_res = get_single_gaps(st, row["receptor_chain"], res_list)
    make_gaps_gly(st, row, gap_res)

    # Mutating residues can change the atom numbering so let's retreive
    # all the atom indices in a separate loop
    for res_num in gap_res + res_list:
        res = st.findResidue(f'{row["receptor_chain"]}:{res_num}')
        if res.hasMissingAtoms():
            print(f'Missing atoms in rec res {row["receptor_chain"]}:{res_num}')
            raise MissingAtomsError
        res_ats.extend(res.getAtomIndices())

    lig_ats = []
    for res_num in range(row["ligand_residue_number"], row["ligand_residue_end"] + 1):
        res_name = f'{row["ligand_chain"]}:{res_num}{row["ligand_residue_inscode"]}'
        try:
            res = st.findResidue(res_name)
        except MissingResiduesError:
            print(f"missing expected residue: {res_name}")
            continue
        lig_ats.extend(res.getAtomIndices())
    # Add in coordinating groups for ions
    if len(lig_ats) == 1:
        asl_str = f"fillres within 3 atom.num {lig_ats[0]}"
        coord_ats = analyze.evaluate_asl(st, asl_str)
        lig_ats.extend([at for at in coord_ats if at not in res_ats])

    # mark the ligand chain
    for at in lig_ats:
        st.atom[at].chain = "l"
    # mark the receptor chain
    for at in res_ats:
        st.atom[at].chain = "A"

    return lig_ats, res_ats


def get_single_gaps(st: Structure, rec_chain: str, res_list: list[str]) -> list[str]:
    """
    Get residues that are not in a list of residues but are
    between two residues which are in the list.

    If we have a single residue which would not normally be included
    which is between two which are, then the capping groups of those
    two are going to overlap because they are both trying to fill in
    the same missing CA (the CA of the gap residue). We need to treat
    these specially so we retrieve them here.

    :param st: structure to analyze
    :param rec_chain: name of receptor chain
    :param res_list: list of residue names which may have single-residue gaps
    :return: list of residues which are single-residue gaps in `res_list`
    """
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


def cap_termini(st: Structure, ligand_env: Structure) -> None:
    """
    Cap termini for extracted receptor chains with NMA or ACE oriented to match
    the positions of neighboring residues in the original structure.

    :param st: original protein structure from which the pocket was extracted
    :param ligand_env: extracted protein receptor (i.e. ligand environment)
    """
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


def main():
    biolip_df = get_biolip_db()
    start_pdb = int(sys.argv[1])
    end_pdb = int(sys.argv[2])
    ligand_env = retreive_ligand_and_env(
        biolip_df, start_pdb=start_pdb, end_pdb=end_pdb
    )


if __name__ == "__main__":
    main()
