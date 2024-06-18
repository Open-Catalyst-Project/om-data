""" Adapted from Architector/development/lig_sampling/Sampling_script_forMeta_step2.ipynb """

import argparse
from typing import Optional

import architector.io_ptable as io_ptable
import mendeleev
import numpy as np
import pandas as pd
from tqdm import tqdm

MAX_N_ATOMS = 120
# Set seed
random_seed = 98745
np.random.seed(random_seed)


def select_metals(metal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make adjustments to metals DataFrame

    1) Remove rare coordination numbers
    2) Add Pm (by duplicating parameters of Sm)
    3) Exclude any metals centers we don't want to consider

    :param metal_df: input metal list to correct
    :return: corrected metal list
    """
    all_fracts = []
    for i, row in metal_df.iterrows():
        fracts = np.array(row["coreCN_counts_CSD"])
        out_fracts = fracts / fracts.sum()
        all_fracts.append(np.round(out_fracts, 3))
    metal_df["coreCN_fracts"] = all_fracts
    # Get rid of CNs found in less than 1 % of cases in CSD # -> Added
    newrows = []
    for i, row in metal_df.iterrows():
        newrow = row.copy()
        if row["coreCN_fracts"].shape[0] > 0:
            save_cn_inds = np.where(row["coreCN_fracts"] > 0.01)[0]
            newrow["coreCNs"] = np.array(row["coreCNs"])[save_cn_inds]
            newrow["coreCN_counts_CSD"] = np.array(row["coreCN_counts_CSD"])[
                save_cn_inds
            ]
            newrow["coreCN_fracts"] = np.array(row["coreCN_fracts"])[save_cn_inds]
            newrow["total_count"] = np.sum(newrow["coreCN_counts_CSD"])
            newrows.append(newrow)
    subset_cn_metal_df = pd.DataFrame(newrows)
    # Add Pm with Sm numbers (neighbors of Sm and Nd) - both Sm and Nd have similar values anyway.
    refrow = subset_cn_metal_df[subset_cn_metal_df.metal == "Sm"].iloc[0]
    subset_cn_metal_df.loc[len(newrows)] = {
        "metal": "Pm",
        "ox": 3,
        "coreCNs": refrow["coreCNs"],
        "coreCN_counts_CSD": refrow["coreCN_counts_CSD"],
        "coreCN_fracts": refrow["coreCN_fracts"],
        "total_count": refrow["total_count"],
    }
    # Omit the Ln
    gen_metal_df = subset_cn_metal_df[
        ~subset_cn_metal_df.metal.isin(io_ptable.lanthanides)
    ]
    gen_metal_df.reset_index(drop=True, inplace=True)
    return gen_metal_df


def sample(
    metal_df: pd.DataFrame,
    ligands_df: pd.DataFrame,
    test: bool = False,
    maxCN: int = 12,
) -> tuple[dict, str]:
    """
    Take samples from the metal DataFrame first, then the ligands DataFrame

    It assumes a flat probability across all metal/ligand combinations.

    We could implement a frqeuency bias filter.

    :param metal_df: Metals dataframe to sample from
    :param ligands_df: Ligands dataframe to sample from
    :param test: Whether to generate test dataframe (faster for Architector)
    :param maxCN: Maximum coordination number to sample
    :return: A row with as much metadata as needed to backtrace how this chemistry was
             sampled and unique identifier for the chemistry sampled
    """
    metal_row = metal_df.sample(1).iloc[0]
    if len(metal_row["coreCNs"]) > 0:  # Ensure there's at least 1 coreCN in list
        cn = np.random.choice(metal_row["coreCNs"], size=1)[0]
    else:  # If not, try from 5-11.
        cn = np.random.choice(np.arange(5, 12, 1), size=1)[0]
    if cn > maxCN:
        cn = np.random.choice(np.arange(2, maxCN + 1, 1), size=1)[0]
    coordsites_left = cn
    natoms_total = 1
    complex_charge = metal_row["ox"]
    ox = metal_row["ox"]
    metal = metal_row["metal"]
    # Use mendeleev to assign spin (assigns highest possible in all cases I've seen)
    spin = mendeleev.__dict__[metal].ec.ionize(ox).unpaired_electrons()
    uid = "{}_ox{}_cn{}_".format(metal, ox, cn)
    architector_input = {
        "core": {"metal": metal, "coreCN": int(cn)},
        "ligands": [],
        "parameters": {
            "metal_ox": ox,
            "metal_spin": spin,
            "assemble_method": "GFN2-xTB",
            "full_method": "GFN2-xTB",
            "n_conformers": 3,  # Will return relaxed 3 lowest-energy XTB conformers per symmetry if distinct enough.
            "n_symmetries": 10,
            "full_graph_sanity_cutoff": 2.0,  # Increasing to loop in more structures where ligand may be falling off
        },
    }
    if test:
        architector_input["parameters"].update(
            {"assemble_method": "UFF", "full_method": "UFF", "n_conformers": 1}
        )
    liguids = []
    lig_dents = []
    lig_charges = []
    lig_coord_atom_types = []
    lig_frequencies = []
    lig_natoms = []
    finished = False
    while not finished:
        tdf = ligands_df[
            ((ligands_df.charge + complex_charge) > -3)
            & ((ligands_df.charge + complex_charge) < 5)  # Filter 1 -> Charge > -3
            & (ligands_df.natoms + natoms_total < MAX_N_ATOMS)  # Filter 2 -> Charge < 5
            & (  # Filter 3 -> Max number of atoms (set at 250 now.)
                coordsites_left - ligands_df.denticity >= 0
            )  # Can fit at the metal surface with remaining coordination sites.
        ]
        # Check that there's any ligands that match the constraints
        if tdf.shape[0] > 0:
            # Sample from the ligands
            add_row = tdf.sample(1, weights=tdf.denticity).iloc[0]
            # Weighting by denticity makes this equal likelihood PER coordination site.
            lig_dict = {"smiles": add_row["smiles"], "coordList": add_row["coordList"]}
            architector_input["ligands"] = architector_input["ligands"] + [lig_dict]
            coordsites_left = coordsites_left - add_row["denticity"]
            complex_charge = complex_charge + add_row["charge"]
            natoms_total = natoms_total + add_row["natoms"]
            lig_dents.append(add_row["denticity"])
            liguids.append(add_row["uid"])
            lig_charges.append(add_row["charge"])
            lig_frequencies.append(add_row["frequency"])
            lig_natoms.append(add_row["natoms"])
            lig_coord_atom_types.append(add_row["coord_atom_types"])
            # Only finish when coordination environment filled.
            if coordsites_left == 0:
                finished = True
        else:
            finished = True

    # Ensure the ligands are sorted for uid for checking for duplication.
    liguids_order = np.argsort(liguids)
    liguids = np.array(liguids)[liguids_order]
    lig_dents = np.array(lig_dents)[liguids_order]
    lig_charges = np.array(lig_charges)[liguids_order]
    lig_frequencies = np.array(lig_frequencies)[liguids_order]
    lig_natoms = np.array(lig_natoms)[liguids_order]
    lig_coord_atom_types = np.array(lig_coord_atom_types)[liguids_order]
    uid = uid + "_".join(liguids)
    sample_row = {
        "label": uid,
        "total_charge": complex_charge,
        "n_atoms_total": natoms_total,
        "metal_coordination_number": cn,
        "metal_ox": ox,
        "metal": metal,
        "ligands": liguids,
        "lig_denticities": lig_dents,
        "lig_charges": lig_charges,
        "lig_frequencies": lig_frequencies,
        "lig_natoms": lig_natoms,
        "lig_coord_atom_types": lig_coord_atom_types,
        "architector_input": architector_input,
    }  # Save the sample and uid
    return sample_row, uid


def create_sample(
    metal_df: pd.DataFrame,
    ligands_df: pd.DataFrame,
    history_uids: Optional[list] = None,
    nsamples: int = 100,
    test: bool = False,
    maxCN: int = 12,
) -> tuple[pd.DataFrame, list]:
    """
    Create Architector inputs for generating metal-organic complexes.

    The inputs are added to a DataFrame so that they can be parsed and run by the generate_structures.py script

    :param metal_df: Metals dataframe to sample from
    :param ligands_df: Ligands dataframe to sample from
    :param history_uids: List of chemistries already sampled to avoid, by default None
    :param nsamples: Number of samples to create in this pass, by default 100
    :param test: Use faster parameters for architector
    :return: Sample chemistries with architector_input to pass to generation script and
             the UIDS of the chemistries sampled up to this point by the sampler routine.
    """
    if history_uids is None:
        history_uids = []
    total = 0
    out_rows = []
    with tqdm(total=nsamples) as pbar:
        while total < nsamples:
            sample_row, uid = sample(
                metal_df=metal_df, ligands_df=ligands_df, test=test, maxCN=maxCN
            )
            if uid not in history_uids:
                total += 1
                history_uids.append(uid)
                out_rows.append(sample_row)
                pbar.update(1)
    dfout = pd.DataFrame(out_rows)
    return dfout, history_uids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outname",
        type=str,
        required=True,
        help="Name to use for output files",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to take",
    )
    parser.add_argument(
        "--history",
        type=str,
        help="Path to file storing previously used samples to avoid duplication",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    metal_df = pd.read_pickle("metal_sample_dataframe.pkl")
    ligands_df = pd.read_pickle("ligand_sample_dataframe.pkl")
    if args.history is not None:
        with open(args.history, "r") as fh:
            history = eval(fh.read())
    else:
        history = None

    gen_metal_df = select_metals(metal_df)
    sdf, history = create_sample(
        metal_df=gen_metal_df,
        ligands_df=ligands_df,
        test=False,
        history_uids=history,
        nsamples=args.n_samples,
    )

    sdf.to_pickle(f"{args.outname}.pkl")
    with open(f"{args.outname}_uids", "w") as fh:
        fh.write(str(history))


if __name__ == "__main__":
    main()
