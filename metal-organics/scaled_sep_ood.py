from architector import convert_io_molecule
from ase.io import read
import os
import argparse
import glob
from functools import partial
from tqdm import tqdm
import multiprocessing as mp

#fname = '/private/home/levineds/distance_scaling_structures/distance_scaling_mca_idx_6471.traj'
#fname = '/private/home/levineds/distance_scaling_structures/distance_scaling_mcc_idx_21967.traj'
def main_loop(fname, output_path):
    atoms = read(fname)
    mol = convert_io_molecule(atoms)
    try:
        trajs = mol.lig_dissociation_sample(
            max_dist=8, # Distance to push ligand away
            steps=21, # Number of steps on trajectory, note that starting point is already done
        )
    except:
        return
    if atoms.info["source"].endswith('.sdf'):
        desc = os.path.splitext(atoms.info["source"])[0]
    elif atoms.info["source"].startswith('ml_mo'):
        desc = os.path.basename(os.path.dirname(atoms.info["source"]))
        desc += f'_{atoms.info["charge"]}_{atoms.info["spin"]}'
    else:
        desc = os.path.basename(os.path.dirname(atoms.info["source"]))
        if desc.startswith("step"):
            desc = os.path.basename(os.path.dirname(os.path.dirname(atoms.info["source"])))

    *rest, charge, spin = desc.split("_")
    for lig_idx, lig in enumerate(trajs):
        for pos_idx, mol2_block in enumerate(lig[1:], start=1):
            mol = convert_io_molecule(mol2_block)
            label = "_".join(rest + [f'lig{lig_idx}', f'dist{pos_idx}', charge, spin]) + ".xyz"
            xyz = mol.write_xyz(os.path.join(output_path, label), writestring=False)

#main_loop(fname, '.')
def main(output_path):
    flist = glob.glob(
        f"/private/home/levineds/distance_scaling_structures/distance_scaling_mcc*.traj"
    )
    scale_fxn = partial(main_loop, output_path=output_path)
    with mp.Pool(60) as pool:
        list(tqdm(pool.imap(scale_fxn, flist), total=len(flist)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.output_path)
