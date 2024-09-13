import glob
import os
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
from schrodinger.structure import StructureReader, StructureWriter
from schrodinger.application.jaguar.utils import mmjag_update_lewis
import json
import sys
print(os.getcwd())
filename = sys.argv[1]
with change_directory(sys.argv[2]):
    # Load JSON file
    with open(f'metadata_{filename}.json', 'r') as file:
        data = json.load(file)
        print(data)
        st_list = []
        charges = data["charges"]
        for i, fh in enumerate(data['species']):
            print(fh+'.pdb')
            for st in StructureReader(fh+".pdb"):
                st.property['i_m_Molecular_charge'] = charges[i]
                mmjag_update_lewis(st)
                st_list.append(st)
        print(st_list)
        with StructureWriter('monomers.maegz') as writer:
             writer.extend(st_list)
