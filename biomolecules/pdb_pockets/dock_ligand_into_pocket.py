import argparse
import glob
import json
import os
import pickle
import random
import subprocess
import tarfile
import tempfile
from urllib.request import urlretrieve

import pandas as pd
from rdkit import Chem
from schrodinger.adapter import to_structure
from schrodinger.comparison import are_conformers
from schrodinger.structure import StructureReader, count_structures
from schrodinger.structutils import build, transform
from schrodinger.rdkit.rdkit_adapter import from_rdkit
from tqdm import tqdm

SCHRO = "/private/home/levineds/desmond/mybuild"


def download_geom():
    tar_name = "rdkit_folder.tar.gz"
    urlretrieve("https://dataverse.harvard.edu/api/access/datafile/4327252", tar_name)
    with tarfile.open(tar_name, "r:gz") as tar:
        tar.extractall()


def load_zinc(zinc_path='zinc_2783k.json'):
    with open(zinc_path, "r") as fh:
        data = json.loads(fh.read())
    zinc_df = pd.DataFrame(data)
    return zinc_df

def load_zinc_leads(zinc_path):
    return glob.glob(os.path.join(zinc_path, '*.sdf'))

def load_geom(geom_path):
    drugs_file = os.path.join(geom_path, "summary_drugs.json")

    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)
    return drugs_summ

def load_chembl(chembl_path):
    st_list = list(StructureReader(chembl_path))
    return st_list

def get_ligand_centroid(st):
    ligand = st.chain["l"].extractStructure()
    at_list = [at.index for at in ligand.atom if at.atomic_number > 1]
    box_center = transform.get_centroid(ligand, at_list)
    return box_center


def remove_ligand(st):
    lig_ats = st.chain["l"].getAtomList()
    if "c" in [ch.name for ch in st.chain]:
        lig_ats.extend(st.chain["c"].getAtomList())
    st.deleteAtoms(lig_ats)


def write_pdbqt(fname, st):
    file_base, file_ext = os.path.splitext(fname)
    pdb_name = fname.replace(file_ext, ".pdb")
    st.write(pdb_name)
    subprocess.run(
        [
            os.path.join(SCHRO, "utilities", "obabel"),
            "-ipdb",
            pdb_name,
            "-opdbqt",
            f"-O{file_base}.pdbqt",
        ]
    )
    os.remove(pdb_name)
    return file_base + ".pdbqt"


def write_zinc_ligand(smi, ligand_name):
    st = to_structure(smi)
    st.generate3dConformation(require_stereo=False)
    build.add_hydrogens(st)
    st.write(ligand_name)


def run_smina(receptor_pdbqt, box_center, ligand_sdf):
    dock_name = (
        f"{os.path.splitext(receptor_pdbqt)[0]}_{os.path.splitext(ligand_sdf)[0]}"
    )
    log_file = f"{dock_name}.log"
    output_sdf = f"{dock_name}.sdf"
    energy_range = 3
    exhaustiveness = 16
    min_rmsd_filter = 1.5
    max_num_poses = 1
    size = 20

    cmd = (
        f"/private/home/levineds/smina/smina.static --cpu 1 -r {receptor_pdbqt} -l {ligand_sdf} "
        f"--center_x {box_center[0]:.4f} --center_y {box_center[1]:.4f} --center_z {box_center[2]:.4f} "
        f"--size_x {size} --size_y {size} --size_z {size} --min_rmsd_filter {min_rmsd_filter} "
        f"--exhaustiveness {exhaustiveness} --energy_range {energy_range} -o {output_sdf} --num_modes {max_num_poses} --log {log_file}"
    )
    subprocess.run(cmd.split())
    os.remove(log_file)
    return output_sdf


def sample_pocket(pdb_path, pocket_df):
    pocket_entry = pocket_df.iloc[random.randint(0, len(pocket_df) - 1)]
    mae_file = pocket_entry.name
    st = StructureReader.read(os.path.join(pdb_path, mae_file))
    return st, mae_file


def run_epik(mae_name):
    out_name = os.path.splitext(mae_name)[0] + "_out.maegz"
    subprocess.run(
        [
            os.path.join(SCHRO, "epikx"),
            mae_name,
            out_name,
            "-ph",
            "7.4",
            "-pht",
            "1.0",
            "-q_hi",
            "2",
            "-q_lo",
            "-2",
            "-ms",
            "3",
            "-NOJOBID",
        ]
    )
    st_list = list(StructureReader(out_name))
    for fname in [mae_name, out_name, os.path.splitext(out_name)[0] + ".log"]:
        os.remove(fname)
    return st_list

def get_charged_states(input_name):
    ligs = run_epik(input_name)
    ligand_basename = os.path.splitext(input_name)[0]
    ligand_fnames = []
    for idx, lig in enumerate(ligs):
        fname = f"{ligand_basename}_ligstate{idx}.sdf"
        lig.write(fname)
        ligand_fnames.append(fname)
    return ligand_fnames

def sample_zinc_leads_ligand(zinc_list):
    zinc_sdf = random.choice(zinc_list)
    total = count_structures(zinc_sdf)
    zinc_reader = StructureReader(zinc_sdf)
    zinc_reader.setIndex(random.randint(1, total))
    zinc_st = next(zinc_reader)
    mae_fname = zinc_st.title + ".maegz"
    zinc_st.write(mae_fname)
    return mae_fname


def sample_zinc_ligand(zinc_df):
    zinc_entry = zinc_df.iloc[random.randint(0, len(zinc_df) - 1)]
    mae_fname = zinc_entry["zincid"] + ".maegz"
    write_zinc_ligand(zinc_entry["SMILES"], mae_fname)
    return mae_fname


def sample_geom_ligand(geom_path, geom_dict):
    drug_key = list(geom_dict)[random.randint(0, len(geom_dict) - 1)]
    drug_pkl_path = os.path.join(geom_path, geom_dict[drug_key]["pickle_path"])
    with open(drug_pkl_path, "rb") as fh:
        mol_dict = pickle.load(fh)
    mol = mol_dict["conformers"][0]["rd_mol"]
    # Check if rdkit mol has the correct charge
    st = to_structure(drug_key)
    mol_st = from_rdkit(mol)
    ligand_basename = f'{mol_dict["conformers"][0]["geom_id"]}'
    if not are_conformers(st, mol_st, use_lewis_structure=False):
        st.generate3dConformation(require_stereo=False)
        st.write(ligand_basename + '.sdf')
    else:
        Chem.MolToMolFile(mol, ligand_basename + ".sdf")
    return ligand_basename + '.sdf'

def sample_chembl_ligand(chembl_list):
    st = random.choice(chembl_list)
    ligand_basename = st.title
    st.write(ligand_basename + '.sdf')
    return ligand_basename + '.sdf'

def dock_randomly(pdb_path, pockets_df, lig_repo, lig_source, geom_path, output_path, done_systems):
    rand_pocket, pocket_name = sample_pocket(pdb_path, pockets_df)
    if lig_source == "geom":
        try:
            rand_lig = sample_geom_ligand(geom_path, lig_repo)
        except KeyError: #GEOM is sometimes missing pickle paths
            return
    elif lig_source == "zinc":
        rand_lig = sample_zinc_ligand(lig_repo)
    elif lig_source == "zinc_leads":
        rand_lig = sample_zinc_leads_ligand(lig_repo)
    elif lig_source == "chembl":
        rand_lig = sample_chembl_ligand(lig_repo)
    
    if any(done.startswith(os.path.splitext(pocket_name)[0] + '_' + os.path.splitext(rand_lig)[0]) for done in done_systems):
        return

    rand_ligs = get_charged_states(rand_lig)

    box_center = get_ligand_centroid(rand_pocket)
    remove_ligand(rand_pocket)

    receptor_name = write_pdbqt(pocket_name, rand_pocket)

    for rand_lig in rand_ligs:
        pock_copy = rand_pocket.copy()
        docked_lig_name = run_smina(receptor_name, box_center, rand_lig)
        docked_lig = StructureReader.read(docked_lig_name)
        build.add_hydrogens(docked_lig)
        os.remove(docked_lig_name)
        for ch in docked_lig.chain:
            ch.name = "l"
        pock_copy.extend(docked_lig)
        save_name = (
            os.path.splitext(os.path.basename(docked_lig_name))[0]
            + f"_{pock_copy.formal_charge}_1.maegz"
        )
        pock_copy.write(os.path.join(output_path, save_name))
        print(os.path.join(output_path, save_name))
        os.remove(rand_lig)
    os.remove(receptor_name)


def main(args):
    pockets_df = pd.read_pickle(args.pockets_pkl_path)
    pockets_df = pockets_df[
        pockets_df["has_binding_info"]
        & ~pockets_df["is_covalently_bound"]
        & pockets_df.index.str.contains("_state0_")
    ]

    if args.lig_source == "zinc":
        lig_repo = load_zinc()
    if args.lig_source == "zinc_leads":
        lig_repo = load_zinc_leads(args.lig_file_path)
    elif args.lig_source == "geom":
        lig_repo = load_geom(args.lig_file_path)
    elif args.lig_source == 'chembl':
        lig_repo = load_chembl(args.lig_file_path)
    done_systems = [os.path.basename(f) for f in glob.glob(os.path.join(args.output_path, '*'))]

    for i in tqdm(range(args.n_to_dock), total=args.n_to_dock):
        dock_randomly(
            args.pdb_path,
            pockets_df,
            lig_repo,
            args.lig_source,
            args.lig_file_path,
            args.output_path,
            done_systems,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--random_seed", type=int, default=142)
    parser.add_argument("--pdb_path", type=str, default=".")
    parser.add_argument("--pockets_pkl_path", type=str, default=".")
    parser.add_argument("--lig_file_path", type=str, default=".")
    parser.add_argument("--lig_source", type=str, default="zinc")
    parser.add_argument("--n_to_dock", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.random_seed)
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        main(args)
