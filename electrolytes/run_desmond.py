import argparse
import contextlib
import csv
import os 
import shutil
from schrodinger.job import queue
from generatesolvent_desmond import generate_solvent_desmond
from generatesystem_desmond import generate_system
from computedensity import compute_density

@contextlib.contextmanager
def chdir(destination):
    # Store the current working directory
    current_dir = os.getcwd()
    
    try:
        # Change to the new directory
        os.chdir(destination)
        yield
    finally:
        # Change back to the original directory
        os.chdir(current_dir)


def get_desmond_cmd(input_name, output_name, job_spec, job_name):
    return ['$SCHRODINGER/utilities/multisim', '-o', output_name, '-mode', 'umbrella', input_name, '-m', job_spec, '-HOST', 'localhost', '-JOBNAME', job_name]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--job_idx", type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    job_dj = queue.JobDJ()

    with open("elytes.csv", "r") as f:
        systems = list(csv.reader(f))
    if args.job_idx is not None:
        job_range = [args.job_idx]
    else:
        job_range = range(1, len(systems))
    
    density = None

    # Define the numbers as a space-separated list
    # Loop through each number
    for job_idx in job_range:
        units = systems[job_idx][3]
        #if units != 'volume':
            #continue

        job_dir = os.path.join(args.output_path, str(job_idx))
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        os.makedirs(job_dir)
        
        if units == 'volume':
            command, directory = generate_solvent_desmond(job_idx, systems, job_dir)
            job_dj.addJob(queue.JobControlJob(command, directory))
            job_dj.run()
            
            cmd = get_desmond_cmd('solvent_system-out.cms', 'solvent_density.cms', 'solvent_multisim.msj', 'density')
            job_dj.addJob(queue.JobControlJob(cmd, job_dir))
            job_dj.run()

            with chdir(job_dir):
                density = compute_density(job_dir)
        command, directory = generate_system(job_idx, systems, job_dir, density)
        job_dj.addJob(queue.JobControlJob(command, directory))
        job_dj.run()

        cmd = get_desmond_cmd('elyte_system-out.cms', 'final_config.cms', 'elyte_multisim.msj', 'final')
        job_dj.addJob(queue.JobControlJob(cmd, job_dir))
        job_dj.run()

if __name__ == "__main__":
    main()
