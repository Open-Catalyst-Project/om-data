""" Adapted from Architector/development/lig_sampling/production_scripts/mpirun.py """

import argparse
import os
import pathlib
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# Note - these are needed at this level as well to force the correct number of threads to xTB.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def calc(input_dict: dict, outpath: str) -> bool:
    """
    Run a subprocess which generates the structure for a given input.

    :param input_dict: parameters for Architector
    :param outpath: path to save results to
    :return: True if function completes
    """
    with open(input_dict["name"] + "_running", "w") as file1:
        file1.write(str(datetime.now()))
    pickle_input = pickle.dumps(input_dict, 0).decode(encoding="latin-1")
    start = time.time()
    try:
        subprocess.check_output(
            ["python", "generate_structures.py", pickle_input, outpath],
            universal_newlines=True,
        )
    except subprocess.CalledProcessError as e:
        end = time.time()
        with open(
            os.path.join(outpath, input_dict["name"] + "_failed.txt"), "w"
        ) as file1:
            file1.write("Total_Time: {}\n".format(end - start))
            file1.write("Failed at : {}\n".format(datetime.now()))
            file1.write("Subprocess error: {}\n".format(e))
        pass
    os.remove(input_dict["name"] + "_running")
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument(
        "--n_workers",
        type=int,
        required=True,
        help="Number of simultaneous jobs to run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Split up the pickle in to batches of this size",
    )
    parser.add_argument(
        "--batch_idx",
        type=int,
        required=True,
        help="Given the batch size specified, run the nth batch",
    )
    parser.add_argument(
        "--outpath",
        type=str,
        required=True,
        help="Path to save completed structures to",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    n_workers = args.n_workers
    batch_idx = args.batch_idx
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)
    # Load input dataframe
    indf = pd.read_pickle(args.input_path)

    # Check the output path to not duplicate
    # Finished/failed architector runs.
    op = pathlib.Path(args.outpath)
    done_list = [
        p.name.replace(".pkl", "").replace("_failed.txt", "") for p in op.glob("*")
    ]
    # Add index as name of job from input dataframe.
    newindf_rows = []
    for i, row in indf.iterrows():
        if str(i) in done_list:
            continue
        inp_dict = row["architector_input"]
        inp_dict["name"] = str(i)
        newindf_rows.append(inp_dict)

    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futs = []
        for d in newindf_rows[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]:
            # print('Submitting: {}'.format(i))
            fut = exe.submit(calc, d, args.outpath)
            futs.append(fut)
        # Track progress of completed jobs.
        for x in tqdm(as_completed(futs), total=len(futs)):
            res = x.result()


if __name__ == "__main__":
    main()
