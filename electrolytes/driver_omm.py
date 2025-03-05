import os
import runmd_omm

row_number = 1

# Only generate system if not restarting (i.e. checkpoint file doesn't exist)
checkpoint_file = os.path.join(f"{row_number}", "md.chk")
if not os.path.exists(checkpoint_file):
    import system_generator_omm
    system_generator_omm.main("csv", file="rpmd_elytes.csv", density=0.5, row=row_number)

# Run simulation, which can be restarted from a checkpoint file
result = runmd_omm.run_simulation(
    pdb_file=f"{row_number}/system.pdb",
    xml_file=f"{row_number}/system.xml", 
    output_dir=f"{row_number}",
    t_final=500.0,
    n_frames=100,
  #  rpmd=True,
  #  num_replicas=32,
    dt=0.002
)