import argparse
import glob
import multiprocessing as mp
import os
import random
from functools import partial
from typing import List

from schrodinger.structure import Structure, StructureReader
from schrodinger.structutils.transform import get_centroid, translate_structure
from tqdm import tqdm

mae_dir = "/checkpoint/levineds/elytes_09_29_2024/results2"


def dilate_distance(st: Structure, scale_factor: float) -> None:
    """
    Dilate the distances between molecules by a scale factor.

    :param st: The structure to dilate
    :param scale_factor: The factor to change the molecule separation by
    """
    system_centroid = get_centroid(st)
    for mol in st.molecule:
        mol_centroid = get_centroid(st, atom_list=mol.getAtomList())
        vec = mol_centroid - system_centroid
        scaled_vec = vec * scale_factor
        translate_vec = scaled_vec - vec
        translate_structure(st, *translate_vec[:3], atom_index_list=mol.getAtomList())


def construct_outname(output_path: str, job: List[str], sf_str: str, spin: str) -> str:
    """
    Create output name for scaled mae file

    :param output_path: Path to output directory
    :param job: Job parameters as a list of strings
    :param sf_str: Scale factor as a string
    :param spin: The desired spin state
    :return: the assembled output file name
    """
    job_num, species, radius, rest = job
    radius = radius.replace("radius_", "")
    *rest, charge = rest.split("_")
    new_basename = "_".join([job_num, species, radius] + rest + [sf_str, charge, spin])
    new_name = os.path.join(output_path, new_basename + ".mae")
    return new_name


def main_loop(mae_dir: str, output_path: str, job: str):
    """
    Main loop for scaling separations between molecules

    :param mae_dir: Location of mae files
    :param output_path: Location to store scaled files
    :param job: Name of a file that contains the structure to scale
    """
    job_num, species, radius, *rest, charge, spin = os.path.splitext(
        os.path.basename(job)
    )[0].split("_")

    mae_name = os.path.join(
        mae_dir,
        job_num,
        species,
        f"radius_{radius}",
        "_".join(rest + [charge, spin]) + ".mae",
    )
    if not os.path.exists(mae_name):
        glob_str = os.path.join(
            mae_dir, job_num, species, f"radius_{radius}", "_".join(rest + [charge])
        )
        try:
            mae_name = glob.glob(glob_str + "_*.mae")[0]
        except IndexError:
            print(job, flush=True)
            return
    st = StructureReader.read(mae_name)
    del st.pbc
    st.title = "1.0"
    scale_factors = [random.uniform(0.7, 0.8), random.uniform(1.3, 1.8)]
    for scale in scale_factors:
        st_copy = st.copy()
        dilate_distance(st_copy, scale)
        sf_str = str(round(scale, 2))
        st_copy.title = sf_str
        new_basename = (
            "_".join([job_num, species, radius] + rest + [sf_str, charge, spin])
            + ".mae"
        )
        new_name = os.path.join(output_path, new_basename)
        st_copy.write(new_name)
        break


def main(mae_dir, output_path):
    with open("3A_list.txt", "r") as fh:
        data = [
            os.path.basename(os.path.dirname(f))
            for f in fh.readlines()
            if f.endswith("step0\n")
        ]
    scale_fxn = partial(main_loop, mae_dir, output_path)
    pool = mp.Pool(60)
    list(tqdm(pool.imap(scale_fxn, data), total=len(data)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--mae_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.mae_path, args.output_path)
