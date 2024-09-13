from schrodinger.application.desmond.packages import traj_util
from math import prod
import contextlib

@contextlib.contextmanager
def change_directory(destination):
    # Store the current working directory
    current_dir = os.getcwd()
    
    try:
        # Change to the new directory
        os.chdir(destination)
        yield
    finally:
        # Change back to the original directory
        os.chdir(current_dir)
import numpy as np
import sys
import contextlib
import os.path

with change_directory(sys.argv[1]):
    if os.path.isfile('solvent_density.cms'):
        _, cms_model, tr = traj_util.read_cms_and_traj('solvent_density.cms')
        fr = tr[-1]
        vol = prod(fr.box.diagonal())
        mass = cms_model.total_weight
        Avog = 6.022e23
        np.savetxt("solventdata.txt",[mass/(Avog*vol*1e-24)])
    else:
        print("Warning! solvent_density.cms cannot be found")
