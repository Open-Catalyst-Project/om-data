import argparse
import glob
import multiprocessing as mp
import os
import random
from functools import partial

from ase.io import read
from schrodinger.application.jaguar.utils import mmjag_reset_connectivity
from schrodinger.application.matsci.aseutils import get_structure
from schrodinger.structutils.analyze import evaluate_asl
from tqdm import tqdm

from scaled_separations import dilate_distance


def remove_metal_bonds(st):
    """
    Our original systems had no bonds to metals except for VO and VO2.
    """
    metals = evaluate_asl(st, "metals")
    for metal in metals:
        for bond in list(st.atom[metal].bond):
            if st.atom[metal].element == "V":
                other_at = bond.otherAtom(metal)
                # Check if the other atom is a terminal oxo
                if other_at.element == "O" and other_at.bond_total == 1:
                    continue
            st.deleteBond(*bond.atom)


def sample_with_spacing(low, high, size, spacing=0.05):
    samples = []
    max_attempts = 1000
    attempts = 0
    while len(samples) < size and attempts < max_attempts:
        cand = random.uniform(low, high)
        attempts += 1
        if any(abs(prev - cand) < spacing for prev in samples):
            continue
        samples.append(cand)
    return sorted(samples)


def main_loop(fname, output_path, idx_type):
    atoms = read(fname)
    if "scaled_sep" in atoms.info["source"]:
        return
    desc = os.path.basename(os.path.dirname(atoms.info["source"]))
    if desc.startswith("step"):
        desc = os.path.basename(os.path.dirname(os.path.dirname(atoms.info["source"])))

    *rest, charge, spin = desc.split("_")
    st = get_structure(atoms)
    st.property["i_m_Molecular_charge"] = atoms.info["charge"]
    try:
        mmjag_reset_connectivity(st)
    except:
        print(atoms.info["source"])
        return
    ats_to_del = [at for at in st.atom if at.atomic_number < 1]
    st.deleteAtoms(ats_to_del)
    remove_metal_bonds(st)
    short_range_factors = sample_with_spacing(0.7, 0.95, 5)
    medium_range_factors = sample_with_spacing(1.05, 1.8, 10)
    if idx_type != "eo":
        long_range_factors = sample_with_spacing(1.8, 3.0, 10)
    else:
        long_range_factors = []
    scale_factors = short_range_factors + medium_range_factors + long_range_factors
    for scale in scale_factors:
        st_copy = st.copy()
        dilate_distance(st_copy, scale)
        sf_str = str(round(scale, 2))
        st_copy.title = sf_str
        new_basename = "_".join(rest + [sf_str, charge, spin]) + ".mae"
        new_name = os.path.join(output_path, new_basename)
        st_copy.write(new_name)


def main(output_path, idx_type):
    flist = glob.glob(
        f"/private/home/levineds/distance_scaling_structures/distance_scaling_{idx_type}_idx_*.traj"
    )
    scale_fxn = partial(main_loop, output_path=output_path, idx_type=idx_type)
    with mp.Pool(60) as pool:
        list(tqdm(pool.imap(scale_fxn, flist), total=len(flist)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--idx_type", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.output_path, args.idx_type)
