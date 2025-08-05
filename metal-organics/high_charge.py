"""
Move COD systems where one of the molecules is highly charged to another directory
"""

import glob
import os
import shutil
from collections import defaultdict


code_dict = defaultdict(list)
for fname in glob.glob('/checkpoint/levineds/crystallographic_open_database/structures/*.mae'):
    code, _, _, charge, suffix = os.path.basename(fname).split('_')
    code_dict[code].append(fname)
for vals in code_dict.values():
    if len(vals) > 2:
        for fname in vals:
            shutil.move(fname, os.path.join(os.path.dirname(fname), 'default', os.path.basename(fname)))
