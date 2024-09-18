import csv
import os 
import shutil
from schrodinger.job import queue
from generatesolvent_desmond import generate_solvent_desmond
from generatesystem_desmond import generate_system
from computedensity import compute_density


def get_desmond_cmd(input_name, output_name, job_spec, job_name):
    return ['$SCHRODINGER/utilities/multisim', '-o', output_name, '-mode', 'umbrella', input_name, '-m', job_spec, '-HOST', 'localhost', '-JOBNAME', job_name]


def main():
    job_dj = queue.JobDJ()

    with open("elytes.csv", "r") as f:
        systems = list(csv.reader(f))
    num_lines = len(systems)-1
    print(num_lines)

    # Define the numbers as a space-separated list
    # Loop through each number
    for job_idx in range(1, num_lines+1):
        units = systems[job_idx][3]
        print(units)
        if units != 'volume':
            continue

        job_dir = str(job_idx)
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        os.makedirs(job_dir)
        command, directory = generate_solvent_desmond(job_idx)
        job_dj.addJob(queue.JobControlJob(command, directory))
        job_dj.run()
        
        cmd = get_desmond_cmd('solvent_system-out.cms', 'solvent_density.cms', 'solvent_multisim.msj', 'density')
        job_dj.addJob(queue.JobControlJob(cmd, job_dir))
        job_dj.run()

        density = compute_density(job_dir)
        command, directory = generate_system(job_idx, density)
        job_dj.addJob(queue.JobControlJob(command, directory))
        job_dj.run()

        cmd = get_desmond_cmd('elyte_system-out.cms', 'final_config.cms', 'elyte_multisim.msj', 'final')
        job_dj.addJob(queue.JobControlJob(cmd, job_dir))
        job_dj.run()

if __name__ == "__main__":
    main()
