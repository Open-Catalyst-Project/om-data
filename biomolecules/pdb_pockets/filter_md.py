import argparse
from tqdm import tqdm
import glob
import os

from schrodinger.application.desmond.packages import traj_util
from schrodinger.application.desmond.packages import traj, topo
from schrodinger.structutils.transform import get_centroid
from schrodinger.structutils.analyze import evaluate_asl
"""
Break structure up by molecule and find the closest CA to the ligand in each molecule
If those distances exceed 5A from the initial structure, discard it.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--batch", type=int)
    return parser.parse_args()

def centroid_dist(st, ca_list, lig_list):
    return np.linalg.norm(get_centroid(st, ca_list) - get_centroid(st, lig_list))

def main():
    args = parse_args()
    dir_list = glob.glob(os.path.join(args.output_path, f"{args.prefix}*"))
    if args.batch is not None:
        file_list = file_list[10000 * args.batch : 10000 * (args.batch + 1)]
    for dirname in tqdm(dir_list):
        fname = os.path.join(dirname, os.path.basename(dirname) + '-out.cms')
        msys_model, cms_model, tr = traj_util.read_cms_and_traj(fname)
        ca_list = evaluate_asl(cms_model, 'at.ptype CA and not res NMA')
        lig_list = cms_model.chain['l'].getAtomList()
        fsys_st = cms_model.fsys_st.copy()
        keep_list = []
        dicard_list = []
        start_centroid_dist = centroid_dist(fsys_st, ca_list, lig_list)
        for fr in tr:
            topo.update_fsys_ct_from_frame_GF(fsys_st, cms_model, fr)
            dist = centroid_dist(fsys_st, ca_list, lig_list)
            if abs(start_centroid_dist - centroid_dist) > 5.0:
                discard_list.append(fsys_st.copy())
            else:
                keep_list.append(fsys_st.copy())
                
