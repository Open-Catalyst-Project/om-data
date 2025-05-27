import glob
import os
from collections import defaultdict
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import json
import numpy as np

def group_pes(flist):
    """
    Group the file list for a given system with multiple scale factors
    """
    grouped_pes = defaultdict(list)
    for fname in flist:
        *rest, sf, charge, spin = os.path.splitext(os.path.basename(fname))[0].split('_')
        try:
            sf = float(sf)
        except ValueError:
            sf = int(sf.replace('dist',''))
        grouped_pes['_'.join(rest + [charge, spin])].append((sf, fname))
    for group in grouped_pes.values():
        group.sort()
    return grouped_pes

def count_sign_changes(arr):
    return np.sum((arr[:-1] * arr[1:]) < 0)

def is_monotonic(arr, threshold=1e-5):
    return np.all(np.diff(arr) >= -threshold) or np.all(np.diff(arr) <= threshold)

def get_base_cases():
    print('getting base cases...')
    flist = glob.glob(
        f"/private/home/levineds/distance_scaling_structures/distance_scaling_*_idx_*.traj"
    )
    base_cases = {}
    for fname in flist:
        atoms = read(fname)
        desc = os.path.basename(os.path.dirname(atoms.info["source"]))
        if desc.startswith("step"):
            desc = os.path.basename(os.path.dirname(os.path.dirname(atoms.info["source"])))
        base_cases[desc] = atoms
    print('done getting base cases...')
    return base_cases

def split_results_by_vertical():
    if not os.path.exists('vertical_breakdown'):
        mae_dir = '/checkpoint/levineds/ood_scaling/'
        all_files = glob.glob('/large_experiments/opencatalyst/foundation_models/data/omol/evals/processed/scaled_sep/*.json')
        biomolecules = []
        closed_elytes = []
        open_elytes = []
        cod_complexes = []
        arch_complexes = []
        vertical_dict = {'biomolecules': biomolecules, 'closed_elytes': closed_elytes, 'open_elytes': open_elytes, 'cod_complexes': cod_complexes, 'arch_complexes': arch_complexes}
        for fname in tqdm(all_files):
            search_name = os.path.basename(fname).replace('.json', '.xyz')
            found = False
            for loc in ('arch_complexes', 'cod_complexes'):
                if os.path.exists(os.path.join(mae_dir, loc, search_name)):
                    vertical_dict[loc].append(fname)
                    found = True
                    break
            if found:
                continue
            search_name = os.path.basename(fname).replace('.json', '.mae')
            for loc in ('biomolecules', 'closed_elytes', 'open_elytes'):
                if os.path.exists(os.path.join(mae_dir, loc, search_name)):
                    vertical_dict[loc].append(fname)
                    break
        with open('vertical_breakdown', 'w') as fh:
            json.dump(vertical_dict, fh)
    else:
        with open('vertical_breakdown', 'r') as fh:
            vertical_dict = json.loads(fh.read())
    return vertical_dict


def determine_valid_pes_curves(vertical_dict)
    if not os.path.exists('good_groups'):
        good_groups = defaultdict(list)
        for vert_name, vert in list(vertical_dict.items()):
            
            groups = group_pes(vert)
            good = 0
            for name, group in tqdm(groups.items(), total=len(groups)):
                energies = []
                s2_dev = []
                factors = []
                for sf, fname in group:
                    if sf < 1.0:
                        continue
                    with open(fname, 'r') as fh:
                        data = json.load(fh)
                    if data['s_squared_dev'] > 1.1:
                        continue
                    energies.append(data['total_energy [Eh]'])
                    s2_dev.append(data['s_squared_dev'])
                    factors.append(sf)
                if len(factors) < 5:
                    continue
                energy_spline = CubicSpline(factors, energies)
                spin_spline = CubicSpline(factors, s2_dev)
                x_smooth = np.linspace(min(factors), max(factors), 100)
                energy_deriv = energy_spline(x_smooth, 1)
                spin_deriv = spin_spline(x_smooth, 1)
                if np.std(energy_deriv) < 0.1 or np.std(spin_deriv) < 1:
                    good_groups[vert_name].append(group)
                    good += 1
                  #  print(f'check out {name}')
                  #  print(good)
                  #  print(factors)
                  #  print(energies)
                  #  print(s2_dev)

        with open('good_groups', 'w') as fh:
            json.dump(good_groups, fh)
    else:
        with open('good_groups', 'r') as fh:
            good_groups = json.loads(fh.read())
    return good_groups

def main()
    vertical_dict = split_results_by_vertical()
    good_groups = determine_valid_pes_curves(vertical_dict)
    for vert, groups in good_groups.items():
        print(len(groups))
        print(sum(len(group) for group in groups))

if __name__ == '__main__':
    main()
