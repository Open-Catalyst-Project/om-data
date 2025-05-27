import argparse
import os
import runmd_omm
import pandas as pd
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", required=True, type=int)
    parser.add_argument("--csv_file", required=True, type=str)
    parser.add_argument("--rpmd", action='store_true')
    parser.add_argument("--n_beads", type=int, default=32)
    parser.add_argument("--droplet",action='store_true')
    parser.add_argument("--stepsize", type=float, default=0.001, help='step size in picoseconds')
    parser.add_argument("--total_time", type=float, default=20, help='total simulation time in nanoseconds')
    return parser.parse_args()

def main(csv_file, row_number, is_rpmd, is_droplet, n_beads, stepsize, total_time):

    # Read the CSV file to get the temperature
    df = pd.read_csv(csv_file)
    temperature = df.iloc[row_number]['temperature']

    # Only generate system if not restarting
    checkpoint_file = os.path.join(f"{row_number}", "md.chk")
    if not os.path.exists(checkpoint_file):
        import system_generator_omm
        system_generator_omm.main("csv", file=csv_file, density=1.0, row=row_number, droplet=is_droplet)

    # Run simulation with temperature from CSV
    ## RPMD settings H2O
    result = runmd_omm.run_simulation(
        pdb_file=f"{row_number}/system.pdb",
        xml_file=f"{row_number}/system.xml", 
        output_dir=f"{row_number}",
        temperature=temperature,  # Use temperature from CSV
        t_final=total_time*1000, #time in ps
        n_frames=100,
        dt=stepsize,  # Fixed timestep for all systems, in ps
        is_droplet=is_droplet,  # Enable droplet mode. False for bulk mode. 
        rpmd=is_rpmd,
        num_replicas=n_beads,
    )

if __name__ == '__main__':
    args = parse_args()
    main(args.csv_file, args.row, args.rpmd, args.droplet, args.n_beads, args.stepsize, args.total_time)

