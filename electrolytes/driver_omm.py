import os
import runmd_omm
import pandas as pd
import sys

row_number = int(sys.argv[1])

# Read the CSV file to get the temperature
df = pd.read_csv("rpmd_elytes.csv")
temperature = df.iloc[row_number]['temperature']

# Only generate system if not restarting
checkpoint_file = os.path.join(f"{row_number}", "md.chk")
if not os.path.exists(checkpoint_file):
    import system_generator_omm
    system_generator_omm.main("csv", file="rpmd_elytes.csv", density=1.0, row=row_number)

# Run simulation with temperature from CSV
result = runmd_omm.run_simulation(
    pdb_file=f"{row_number}/system.pdb",
    xml_file=f"{row_number}/system.xml", 
    output_dir=f"{row_number}",
    temperature=temperature,  # Use temperature from CSV
    t_final=250.0*1000, #time in ps
    n_frames=100,
    dt=0.001  # Fixed timestep for all systems, in ps
    rpmd=True,
    num_replicas=32,
)
