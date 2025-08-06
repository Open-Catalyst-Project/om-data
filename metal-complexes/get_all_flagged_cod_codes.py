import os
import glob

with open('flagged','r') as fh:
    data = [f.split()[0] for f in fh.readlines()]

cod_codes = {os.path.basename(f).split('_')[0] for f in data}
print(len(cod_codes))
dirname = os.path.dirname(data[0])
print(dirname)
delete_list = [f for f in glob.glob(os.path.join(dirname, '*.mae')) if os.path.basename(f).split('_')[0] in cod_codes]
print(len(delete_list))

with open('redo_codes', 'w') as fh:
    fh.writelines([f+'\n' for f in cod_codes])

with open('delete_flagged', 'w') as fh:
    fh.writelines([f+'\n' for f in delete_list])
