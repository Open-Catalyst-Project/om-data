import h5py
import numpy as np
from gpsts.geodesic import construct_geodesic_path
from ase import Atoms
from ase.io import write
import random
import os, sys, pickle

data=pickle.load(open("/Users/samuelblau/Downloads/aimnet_energy.p", "rb"))
print(len(data.keys()))
ind = 0
for _ in data.keys():
    element=data[_]["element"]
    reactant=Atoms(symbols=element, positions=data[_]["reactant"])
    product=Atoms(symbols=element, positions=data[_]["product"])
    TS=Atoms(symbols=element, positions=data[_]["TS"])
    energy=data[_]["energy"]
    path = construct_geodesic_path(reactant, TS, nimages=10)
    path += construct_geodesic_path(TS, product, nimages=10)[1:]
    for image_idx, image in enumerate(path):
        if image_idx < 4 or image_idx >= len(path) - 4:
            continue
        write(f'{ind}_{image_idx}_0_2.xyz', image)
        if random.random() < 0.1:
            write(f'{ind}_{image_idx}_1_3.xyz', image)
        elif random.random() < 0.2:
            write(f'{ind}_{image_idx}_-1_3.xyz', image)
            
    ind += 1
    if ind > 3:
        break