import pandas as pd
import numpy as np
import random
from more_itertools import collapse
from collections import Counter, defaultdict
import os


def get_system_list(archive_name):
    with open(f'/private/home/levineds/job_accounting/metal_organics/archives_{archive_name}.txt') as fh:
        data = fh.readlines()
    system_list = list({system.split('/')[-3] for system in data})
    return system_list


def get_steps_list(archive_name, offset=-3):
    with open(f'/private/home/levineds/job_accounting/metal_organics/archives_{archive_name}.txt') as fh:
        data = fh.readlines()
    system_list = [tuple(system.split('/')[offset:-1]) for system in data]
    return system_list

def group_list(overall_system_list):
    grouped_sys_list = defaultdict(list)
    for sys in overall_system_list:
        grouped_sys_list[int(sys[0].split('_')[0])].append(sys)
    return grouped_sys_list
    
def update_metal_lig_dict(df, grouped_sys_list):
    for key, sys_list in grouped_sys_list.items():
        metal = df.iloc[key]['metal']
        ligands = df.iloc[key]['ligands']
        for lig in ligands:
            metal_lig_dict[(metal, lig)]['count'] += len(sys_list)
            metal_lig_dict[(metal, lig)]['systems'].append(key)

df = pd.read_pickle('/large_experiments/opencatalyst/foundation_models/data/omol/metal_organics/MO_1m.pkl')
system_list2 = defaultdict(list)
large_system_list = set()
duplicates=[]
for i in ('060124','060424','062424','070324','072324','incomplete_060424'):
    lst = get_system_list(i)
    steps = get_steps_list(i)
    duplicates.extend(large_system_list.intersection(steps))
    large_system_list.update(steps)
    for sys in lst:
        system_list2[sys].append(i)
grouped_large_list = group_list(large_system_list)

df_small = pd.read_pickle('/large_experiments/opencatalyst/foundation_models/data/omol/metal_organics/MO_1m_small.pkl')
small_system_list = set()
for i in ('061824',):
    small_system_list.update(get_steps_list(i))
grouped_small_list = group_list(small_system_list)

df_H = pd.read_pickle('/large_experiments/opencatalyst/foundation_models/data/omol/metal_organics/hydride_18200.pkl')
hydride_system_list = set()
for i in ('hydrides',):
    hydride_system_list.update(get_steps_list(i, offset=-2))
grouped_hydride_list = group_list(hydride_system_list)

df_ln = pd.read_pickle('/large_experiments/opencatalyst/foundation_models/data/omol/metal_organics/MO_Ln_255k.pkl')
ln_system_list = set()
for i in ('082524',):
    ln_system_list.update(get_steps_list(i))
grouped_ln_list = group_list(ln_system_list)

metal_lig_dict = defaultdict(lambda: {'count': 0, 'systems':[]})
update_metal_lig_dict(df, grouped_large_list)
update_metal_lig_dict(df_small, grouped_small_list)
update_metal_lig_dict(df_H, grouped_hydride_list)
update_metal_lig_dict(df_ln, grouped_ln_list)

sampled_dict = random.sample(list(metal_lig_dict), 65)

np.sum(metal_lig_dict[key]['count'] for key in sampled_dict[:40])
