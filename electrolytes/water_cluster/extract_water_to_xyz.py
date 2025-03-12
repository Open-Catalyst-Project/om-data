import os

import ase.io
import h5py
from ase import Atoms
from tqdm import tqdm

output_dir = "/checkpoint/levineds/water_clusters"


def extract_h5py(h5_name, start, output_dir):
    atom_list = [8] * 70 + [1] * 140
    with h5py.File(h5_name) as h5:
        data = h5["water"]
        for idx, coords in tqdm(
            enumerate(data["conformations"], start=start), total=50000
        ):
            atoms = Atoms(atom_list, positions=coords * 10.0)
            ase.io.write(
                os.path.join(output_dir, f"water_cluster_{idx}_0_1.xyz"), atoms, "xyz"
            )


for fname, start in zip(*(("water70-1.hdf5", "water70-2.hdf5"), (0, 50000))):
    extract_h5py(fname, start, output_dir)
