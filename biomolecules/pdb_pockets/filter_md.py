import argparse
import glob
import multiprocessing as mp
import os
from functools import partial

import numpy as np
from schrodinger.application.desmond.packages import topo, traj, traj_util
from schrodinger.structure import StructureWriter
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.structutils.rmsd import superimpose
from schrodinger.structutils.transform import get_centroid
from tqdm import tqdm

"""
Break structure up by molecule and find the closest CA to the ligand in each molecule
If those distances exceed 5A from the initial structure, discard it.
"""

N_FRAMES = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--batch", type=int)
    return parser.parse_args()


def centroid_dist(st, ca_list, lig_list):
    return np.linalg.norm(get_centroid(st, ca_list) - get_centroid(st, lig_list))


def remove_similar_frames(frame_list):
    at_list = list(range(1, frame_list[0].atom_total + 1))
    final_frames = frame_list[0:1]
    for frame in frame_list[1:]:
        if superimpose(frame_list[0], at_list, frame.copy(), at_list) > 1.0:
            final_frames.append(frame)
    return final_frames


def subsample_frames(st_list, n_frames):
    frame_list = []
    frame_interval = len(st_list) // n_frames
    if frame_interval < 1:
        frame_list = st_list
    else:
        frame_list = st_list[::frame_interval][:n_frames]
    return frame_list

def extract_frames(dirname, multi=True):
    fname = os.path.join(dirname, os.path.basename(dirname) + "-out.cms")
    try:
        msys_model, cms_model, tr = traj_util.read_cms_and_traj(fname)
    except:
        return
    ca_list = evaluate_asl(cms_model, "at.ptype CA and not res NMA")
    try:
        lig_list = cms_model.chain["l"].getAtomList()
    except KeyError:
        print(fname)
        return
    fsys_st = cms_model.fsys_ct.copy()
    keep_list = []
    start_centroid_dist = None
    for fr in tr:
        topo.update_fsys_ct_from_frame_GF(fsys_st, cms_model, fr)
        dist = centroid_dist(fsys_st, ca_list, lig_list)
        if start_centroid_dist is None:
            start_centroid_dist = dist
        if abs(start_centroid_dist - dist) < 5.0:
            keep_list.append(fsys_st.copy())
    keep_list = remove_similar_frames(keep_list)
    if multi:
        frames = subsample_frames(keep_list, N_FRAMES)
    else:
        frames = keep_list[-1:]
    return frames

def process_md(dirname, output_path):
    frames = extract_frames(dirname)
    if frames is None:
        return
    for idx, frame in enumerate(frames):
        *basename, charge, spin = os.path.splitext(os.path.basename(dirname))[0].split("_")
        name = f'{"_".join(basename)}_frame{idx}_{charge}_{spin}.mae'
        frame.write(os.path.join(output_path, "frames", name))
    return dirname


def main():
    args = parse_args()
    dir_list = sorted(glob.glob(os.path.join(args.output_path, f"{args.prefix}*")))
    already_processed = set()
    for proc_list in glob.glob("processed_traj_list*.txt"):
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

    processed_fname = f"processed_traj_list_{args.prefix}"
    if args.batch is not None:
        processed_fname += f"_{args.batch}"
    processed_fname += ".txt"
    with open(processed_fname, "w") as fh:
        fh.writelines((line + "\n" for line in processed_traj))


if __name__ == "__main__":
    main()
