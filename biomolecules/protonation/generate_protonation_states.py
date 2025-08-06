import argparse
import glob
import multiprocessing as mp
import os
import subprocess
from collections import defaultdict

import pandas as pd
from schrodinger.adapter import to_structure
from schrodinger.application.jaguar.packages.shared import \
    uniquify_with_comparison
from schrodinger.comparison import are_conformers
from schrodinger.structure import StructureReader, StructureWriter
from tqdm import tqdm


def parallel(inp):
    title, smiles = inp
    st = to_structure(smiles)
    st.title = title
    st.generate3dConformation(require_stereo=False)
    return st


def create_input_structures():
    st_list = []
    ## Download the SI from the Epik7 paper
    for csv_name in glob.glob("EpikSI/*"):
        with open(csv_name, "r") as fh:
            df = pd.read_csv(fh)
        print(csv_name)
        with mp.Pool(60) as pool:
            mini_list = list(
                tqdm(pool.imap(parallel, zip(df["Title"], df["SMILES"])), total=len(df))
            )
        st_list.extend(mini_list)

    st_list = uniquify_with_comparison(
        st_list, are_conformers, use_lewis_structure=False
    )

    print(len(st_list))
    with StructureWriter("protonation_structures.mae") as writer:
        writer.extend(st_list)


def run_epik():
    # subprocess.run(['$SCHRODINGER/epikx', 'protonation_structures.mae', 'protonated_structures.maegz', '-ms', '3', '-p', '0.00001', '-JOBNAME', 'epik_pKa', '-HOST', 'localhost', '-gpu', '-retain_i'])
    subprocess.run(
        [
            "$SCHRODINGER/epikx",
            "protonation_structures.mae",
            "protonated_structures.maegz",
            "-ms",
            "3",
            "-p",
            "1e-10",
            "-JOBNAME",
            "epik_pKa",
            "-HOST",
            "localhost",
            "-gpu",
            "-retain_i",
        ]
    )


def extract_final_states(output_path):
    new_st_list = list(StructureReader("protonated_structures.maegz"))
    grouped_sts = defaultdict(list)
    for st in new_st_list:
        grouped_sts[st.title].append(st)

    for key, val in grouped_sts.items():
        new_vals = uniquify_with_comparison(
            val, are_conformers, use_lewis_structure=False
        )
        if len(new_vals) > 1:
            grouped_sts[key] = new_vals

    for key, vals in grouped_sts.items():
        for idx, st in enumerate(vals):
            st.write(
                os.path.join(output_path, f"{key}_state_{idx}_{st.formal_charge}_1.mae")
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # create_input_structures()
    # run_epik()
    extract_final_states(args.output_path)
