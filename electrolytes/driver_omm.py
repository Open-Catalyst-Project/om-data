import os
import runmd_omm
import json
import pandas as pd

row_number = 615
csv_file = "omm-elytes.csv"
# Read the CSV file to get the temperature
df = pd.read_csv(csv_file)
temperature = df.iloc[row_number]['temperature']

# Only generate system if not restarting
checkpoint_file = os.path.join(f"{row_number}", "md.chk")
if not os.path.exists(checkpoint_file):
    import system_generator_omm
    system_generator_omm.main("csv", file=csv_file, density=1.0, row=row_number, droplet=False) # Set droplet to True for droplet mode

# Run simulation with temperature from CSV
result = runmd_omm.run_simulation(
    pdb_file=f"{row_number}/system.pdb",
    xml_file=f"{row_number}/system.xml", 
    output_dir=f"{row_number}",
    temperature=temperature,  # Use temperature from CSV
    t_final=250.0*10, #time in ps
    n_frames=5000,
    dt=0.001,  # Fixed timestep for all systems, in ps
    is_droplet=False#,  # Enable droplet mode. False for bulk mode. 
    #num_replicas=32, # Number of replicas for RPMD
    #rpmd=True # Enable RPMD
)