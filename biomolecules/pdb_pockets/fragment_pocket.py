import argparse
import glob
import multiprocessing as mp
import os
from functools import partial

from schrodinger.structutils import measure
from tqdm import tqdm

from filter_md import extract_frames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--md_path", default=".")
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--batch", type=int)
    return parser.parse_args()


def fragment_pocket(st):
    lig = st.chain["l"].extractStructure()
    st.deleteAtoms(st.chain["l"].getAtomIndices())
    fragment = None
    for i in range(2):
        dist, st_at, lig_at = measure.get_shortest_distance(st, st2=lig)
        chain = st.atom[st_at].getMolecule().extractStructure()
        if fragment is None:
            fragment = chain
        else:
            fragment.extend(chain)
        st.deleteAtoms(st.atom[st_at].getMolecule().getAtomIndices())
        if st.atom_total == 0 or len(fragment.residue) > 5:
            break
    fragment.extend(lig)
    return fragment


def process_md(dirname, output_path):
    frames = extract_frames(dirname, multi=False)
    if frames is None:
        return
    fragment = fragment_pocket(frames[0])
    *basename, charge, spin = os.path.splitext(os.path.basename(dirname))[0].split("_")
    name = f'{"_".join(basename)}_{fragment.formal_charge}_1.maegz'
    fragment.write(os.path.join(output_path, "frames", name))
    return dirname


def main():
    args = parse_args()
    dir_list = sorted(glob.glob(os.path.join(args.md_path, f"{args.prefix}*")))
    already_processed = set()
    for proc_list in glob.glob("processed_frag_list*.txt"):
        with open(proc_list, "r") as fh:
            contents = [line.strip() for line in fh.readlines()]
        already_processed.update(contents)
    dir_list = [f for f in dir_list if f not in already_processed]
    print(len(dir_list))
    if args.batch is not None:
        dir_list = dir_list[1000 * args.batch : 1000 * (args.batch + 1)]
    processed_traj = []

    pool = mp.Pool(60)
    proc_md = partial(process_md, output_path=args.output_path)
    processed_traj = set(tqdm(pool.imap(proc_md, dir_list), total=len(dir_list)))
    processed_traj -= {None}

    processed_fname = f"processed_frag_list_{args.prefix}"
    if args.batch is not None:
        processed_fname += f"_{args.batch}"
    processed_fname += ".txt"
    with open(processed_fname, "w") as fh:
        fh.writelines((line + "\n" for line in processed_traj))


if __name__ == "__main__":
    main()
