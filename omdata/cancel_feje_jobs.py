import subprocess
import multiprocessing as mp
from tqdm import tqdm
import os
import json

#with open('prot_core2_with_tm.txt', 'r') as fh:
#    prot_core_jobs = [os.path.basename(f.strip()) for f in fh.readlines()] 
with open('/checkpoint/levineds/dimers/inputs/paths_list.txt', 'r') as fh:
    jobs = [os.path.basename(f.strip()) for f in fh.readlines()] 
#with open('/checkpoint/levineds/ood_pdb_pockets/md/300K/bioactive_confs/paths.txt', 'r') as fh:
    #tm_jobs = [os.path.splitext(os.path.basename(f.strip()))[0] for f in fh.readlines()] 
#    tm_jobs = {os.path.basename(f.strip()) for f in fh.readlines()}
#
#with open('/private/home/levineds/prot_core2_job_ids.json', 'r') as fh:
#    core_job_id_dict = json.loads(fh.read())
#with open('/private/home/levineds/prot_interface2_job_ids.json', 'r') as fh:
#    interface_job_id_dict = json.loads(fh.read())
with open('/private/home/levineds/job_ids.json', 'r') as fh:
    job_id_dict = json.loads(fh.read())

jobs_to_delete = [job_id_dict[key] for key in jobs if key in job_id_dict]
#jobs_to_delete = list(tm_job_id_dict.values())
#jobs_to_delete.extend([interface_job_id_dict[key] for key in prot_interface_jobs])

print(len(jobs_to_delete))
def feje_del(job_id):
    res = subprocess.run(['feje', 'cancel-job', '--job-id', job_id], capture_output=True)
    #res = subprocess.run(['feje', 'collect-results', '--job-id', job_id, '--timeout', '0', '--delete-from-s3', '--skip-download'], capture_output=True)
    if b"state is JobState.FINISHED"  in res.stderr:
        return job_id

with mp.Pool(60) as pool:
    done_jobs = set(tqdm(pool.imap(feje_del, jobs_to_delete), total = len(jobs_to_delete)))
done_jobs -= {None}

rev_dict = {v:k for k,v in job_id_dict.items()}
done_jobnames = [rev_dict[job_id] for job_id in done_jobs]
with open('done_jobs.txt', 'w') as fh:
    fh.writelines([f+'\n' for f in done_jobnames])
