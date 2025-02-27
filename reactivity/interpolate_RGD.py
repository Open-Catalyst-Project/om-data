import h5py
import numpy as np
from gpsts.geodesic import construct_geodesic_path
from ase import Atoms
from ase.io import write

hf = h5py.File('/private/home/levineds/reactivity/RGD_interpolation/RGD1_CHNO.h5', 'r')
num2element = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F'}

for rxn_idx, rxn in hf.items():
    elements = [num2element[Ei] for Ei in np.array(rxn.get('elements'))]    
    TS_G = np.array(rxn.get('TSG'))
    R_G = np.array(rxn.get('RG'))
    P_G = np.array(rxn.get('PG'))
    reactant = Atoms(symbols=elements, positions=R_G)
    ts = Atoms(symbols=elements, positions=TS_G)
    product = Atoms(symbols=elements, positions=P_G)
    path = construct_geodesic_path(reactant, ts, nimages=10)
    path += construct_geodesic_path(ts, product, nimages=10)[1:]
    for image_idx, image in enumerate(path):
        # All these examples are neutral and closed shell
        write(f'{rxn_idx}_{image_idx}_0_1.xyz', image)
