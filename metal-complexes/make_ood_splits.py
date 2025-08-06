import random
from collections import defaultdict

import numpy as np
import pandas as pd


def get_steps_list(archive_name, offset=-3):
    """
    Get a list of individual optimization steps, i.e. (system_id_geometry_charge_spin, step<n>)
    """
    with open(
        f"/private/home/levineds/job_accounting/metal_organics/archives_{archive_name}.txt"
    ) as fh:
        data = fh.readlines()
    system_list = [tuple(system.split("/")[offset:-1]) for system in data]
    return system_list


def group_list(overall_system_list):
    """
    Group list of steps by the system_id (all steps and geometries with the same id have the same metal-ligands pairs)
    """
    grouped_sys_list = defaultdict(list)
    for sys in overall_system_list:
        grouped_sys_list[int(sys[0].split("_")[0])].append(sys)
    return grouped_sys_list


def update_metal_lig_dict(df, metal_lig_dict, grouped_sys_list, df_name):
    """
    Update a given dict with the metal-ligand combos of the systems in the `grouped_sys_list`
    """
    for key, sys_list in grouped_sys_list.items():
        metal = df.iloc[key]["metal"]
        ligands = df.iloc[key]["ligands"]
        for lig in ligands:
            metal_lig_dict[(metal, lig)]["count"] += len(sys_list)
            metal_lig_dict[(metal, lig)]["systems"].append((df_name, key))


def get_metal_ligs_in_df(df):
    """
    Get all the metal-ligand combos as a dict from a given dataframe
    """
    metal_lig_dict = defaultdict(lambda: {"count": 0, "systems": []})
    for idx, row in df.iterrows():
        metal = row["metal"]
        ligands = row["ligands"]
        for lig in ligands:
            metal_lig_dict[(metal, lig)]["systems"].append(idx)
    return metal_lig_dict


def get_best_version(options):
    """
    Pick a "best" version of the data for duplicated datapoints
    """
    hierarchy = ("low_spin_241118", "072324", "070324", "062424", "060424", "060124", "incomplete_060424", "failed_060424")
    return min(options, key=lambda x: hierarchy.index(x))


def dedup(dup_dict):
    """
    Make a list of systems to discard that are either"
    1) singlets computed with the wrong flag
    2) duplicates of another datapoint (where we keep one that we like the best)
    """
    batch_dict = {
        "failed_060424": "failed_060424",
        "temp_ls": "outputs_low_spin_241118",
    }
    discard = []
    for key, val in dup_dict.items():
        if key[0].split("_")[-1] == "1":
            if "060124" in val:
                val.remove("060124")
                discard.append(f'outputs_060124/{"/".join(key)}')
        keeper = get_best_version(val)
        discard.extend(
            [
                f'{batch_dict.get(version, f"outputs_{version}")}/{"/".join(key)}'
                for version in val if version != keeper
            ]
        )
    return discard


def main():
    metal_lig_dict = defaultdict(lambda: {"count": 0, "systems": []})

    large_system_list = defaultdict(list)
    large_steps_list = set()
    duplicates = []
    for batch_name in ("060124", "060424", "062424", "070324", "072324", "incomplete_060424", "failed_060424", "low_spin_241118"):
        steps = get_steps_list(batch_name)
        duplicates.extend(large_steps_list.intersection(steps))
        large_steps_list.update(steps)
        for step in steps:
            large_system_list[step].append(batch_name)
    grouped_large_list = group_list(large_steps_list)
    df_large = pd.read_pickle(
        "/large_experiments/opencatalyst/foundation_models/data/omol/metal_organics/MO_1m.pkl"
    )
    update_metal_lig_dict(df_large, metal_lig_dict, grouped_large_list, "large")

    dups = {k: v for k, v in large_system_list.items() if len(v) > 1}
    discard = dedup(dups)
    with open("discard_list.txt", "w") as fh:
        fh.write(str(discard))  ## IDK, how do you want to disk a list
    print(len(discard))
    print(len(duplicates))

    small_steps_list = set()
    for batch_name in ("061824",):
        small_steps_list.update(get_steps_list(batch_name))
    grouped_small_list = group_list(small_steps_list)
    df_small = pd.read_pickle(
        "/large_experiments/opencatalyst/foundation_models/data/omol/metal_organics/MO_1m_small.pkl"
    )
    update_metal_lig_dict(df_small, metal_lig_dict, grouped_small_list, "small")

    hydride_steps_list = set()
    for batch_name in ("hydrides",):
        hydride_steps_list.update(get_steps_list(batch_name, offset=-2))
    grouped_hydride_list = group_list(hydride_steps_list)
    df_H = pd.read_pickle(
        "/large_experiments/opencatalyst/foundation_models/data/omol/metal_organics/hydride_18200.pkl"
    )
    update_metal_lig_dict(df_H, metal_lig_dict, grouped_hydride_list, "hydride")

    ln_steps_list = set()
    for batch_name in ("082524",):
        ln_steps_list.update(get_steps_list(batch_name))
    grouped_ln_list = group_list(ln_steps_list)
    df_ln = pd.read_pickle(
        "/large_experiments/opencatalyst/foundation_models/data/omol/metal_organics/MO_Ln_255k.pkl"
    )
    update_metal_lig_dict(df_ln, metal_lig_dict, grouped_ln_list, "ln")

    random.seed(628443)
    sampled_dict = random.sample(sorted(metal_lig_dict), 50)

    print(len(metal_lig_dict))
    print(np.sum(metal_lig_dict[key]["count"] for key in sampled_dict))
    print(sampled_dict)

    df_dict = {"large": set(), "small": set(), "hydride": set(), "ln": set()}

    for key in sampled_dict:
        systems = metal_lig_dict[key]["systems"]
        for arch_run, sys in systems:
            df_dict[arch_run].add(sys)
    with open("ood_splits.txt", "w") as fh:
        fh.write(str(df_dict))

    df_ml = pd.read_pickle(
        "/large_experiments/opencatalyst/foundation_models/data/omol/metal_organics/metal_organics_MS_1M.pkl"
    )
    ml_dict = get_metal_ligs_in_df(df_ml)
    ml_banned = set()
    print("banned from ML")
    for key in sampled_dict:
        ml_banned.update(ml_dict[key]["systems"])
    with open("ood_ml.txt", "w") as fh:
        fh.write(str(ml_banned))


if __name__ == "__main__":
    main()

##
##sampled_dict = [('Yb', np.str_('F[CH-]F1')), ('Bi', np.str_('OC(=O)c1ccc([S-])cc17')), ('Li', np.str_('O=C1[N-]S(=O)(=O)c2ccccc122')), ('Cr', np.str_('CC(C)(C)O[Si]([S-])(OC(C)(C)C)OC(C)(C)C4,6')), ('V', np.str_('[O-]N1C=CC=CC1=S0,7')), ('Pm', np.str_('CC(=NN=C([O-])c1ccccc1)c1ccccn12,5,17')), ('Nd', np.str_('O=[As](c1ccccc1)(c1ccccc1)c1ccccc10')), ('Cr', np.str_('c1ccc(cc1)[N-]c1ccccc1N=Nc1ccccn16,13,20')), ('Hf', np.str_('O=C1[N-]S(=O)(=O)c2ccccc122')), ('K', np.str_('CC(C)c1cccc(C(C)C)c1[N-]c1ccccc1F12,19')), ('Tl', np.str_('Cl[Ge](Cl)Cl1')), ('W', np.str_('CC(C)=O3')), ('Dy', np.str_('S1N=C2c3cccnc3c3ncccc3C2=N17,10')), ('Nd', np.str_('O=N(=O)c1cc(c([O-])c(c1)N(=O)=O)N(=O)=O7')), ('Sm', np.str_('N1C=NC(=N1)c1cnccn12,10')), ('Yb', np.str_('[Se-][Te][Te][Se-]0,3')), ('Mn', np.str_('Brc1ccncc14')), ('Ga', np.str_('O=C1[N-]C(=O)c2ccccc122')), ('Y', np.str_('CCC1=NN=C(N)S14')), ('Ca', np.str_('Cc1cc(C(=O)c2ccccc2)c([O-])c(c1)C(=NCC[Te]c1ccccc1)c1ccccc113,17')), ('Bi', np.str_('OC(=O)c1ccc(cc1)N=Cc1ccccn19,16')), ('Nd', np.str_('NC(N)=O3')), ('Ag', np.str_('CCN(CC)C(C)=N7')), ('Pb', np.str_('Cc1cccc(n1)c1cccc(n1)c1cccc(C)n16,12,19')), ('Rb', np.str_('CN1CC(=O)N=C1N5')), ('Y', np.str_('[Se]=P([Se-])(c1ccccc1)c1ccccc10,2')), ('Al', np.str_('c1cnn(c1)B(n1cccn1)(n1cccn1)n1cccn12,10,15')), ('Nd', np.str_('CC1=CC(=NN1)C4')), ('Ta', np.str_('[Se-]c1ccncc10')), ('Lu', np.str_('CO[C-]=O2')), ('Tm', np.str_('O=S(c1ccccc1P(c1ccccc1)c1ccccc1)c1ccccc1P(c1ccccc1)c1ccccc11,8,27')), ('Gd', np.str_('CN(P(c1ccccc1)c1ccccc1)P(c1ccccc1)c1ccccc12,15')), ('Fe', np.str_('O=C1[N-]C(=O)c2ccccc122')), ('Na', np.str_('CN1N(C(=O)C=C1C)c1ccccc14')), ('Ba', np.str_('Cc1ccc(Br)nc15,6')), ('Gd', np.str_('OC(C(=O)[O-])(c1ccccc1)c1ccccc10,4')), ('La', np.str_('[O-]N=C(C#N)C#N0')), ('Y', np.str_('[S-]c1ccccc10')), ('Nd', np.str_('N1C=CN=C1C1=NC=CN13,6')), ('Cd', np.str_('O=O0')), ('Yb', np.str_('CC(C)(C)[CH-2]4')), ('Cr', np.str_('CCNC([S-])=NN=C(C)c1ccccn14,6,14')), ('Dy', np.str_('[S]#[C-]1')), ('Nb', np.str_('C[Sb](C)Cc1ccccc1C[Sb](C)C1,11')), ('Pm', np.str_('CC[Sb](CC)CC2')), ('Nd', np.str_('Nc1ccc(cc1)S(=O)(=O)[N-]c1ncccn110,16')), ('Ta', np.str_('CSC1=C([S-])SC(=S)S14')), ('Os', np.str_('CN(C)C(C)=O5')), ('Li', np.str_('N#S(N=S(=NS(#N)(c1ccccc1)c1ccccc1)(c1ccccc1)c1ccccc1)(c1ccccc1)c1ccccc10,6')), ('Mo', np.str_('COc1c2Cc3cc(cc(Cc4cc(cc(Cc5cc(cc(Cc1cc(c2)C(C)(C)C)c5[O-])C(C)(C)C)c4OC)C(C)(C)C)c3[O-])C(C)(C)C1,32,38,45'))]
