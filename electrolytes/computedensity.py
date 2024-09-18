import os
from math import prod

import numpy as np
from schrodinger.application.desmond.packages import traj_util


def compute_density(job_dir):
    if os.path.isfile('solvent_density.cms'):
        _, cms_model, tr = traj_util.read_cms_and_traj('solvent_density.cms')
        fr = tr[-1]
        vol = prod(fr.box.diagonal())
        mass = cms_model.total_weight
        Avog = 6.022e23
        density = mass/(Avog*vol*1e-24)
    else:
        raise RuntimeError("Warning! solvent_density.cms cannot be found")
    return density
