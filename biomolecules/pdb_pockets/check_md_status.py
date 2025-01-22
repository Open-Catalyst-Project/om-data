import os

from tqdm import tqdm

temp = "300K"


def reprod_failure(fname):
    log = fname.replace("-out.cms", "_multisim.log")
    if os.path.exists(log):
        with open(log, "r") as fh:
            if "Stage 7 not run" in fh.read():
                return True
    return False


with open(f"{temp}_paths_list.txt", "r") as fh:
    todo = [fn.strip() for fn in fh.readlines()]

still_todo = []
for fn in tqdm(todo):
    outfile = fn.replace("pdb_pockets", f"pdb_pockets/md/{temp}")
    outfile = outfile.replace(".maegz", "")
    outfile += "/" + os.path.basename(outfile) + "-out.cms"
    if not os.path.exists(outfile) and not reprod_failure(outfile):
        still_todo.append(fn)

with open(f"{temp}_paths_list_restart.txt", "w") as fh:
    fh.writelines([fn + "\n" for fn in still_todo])
