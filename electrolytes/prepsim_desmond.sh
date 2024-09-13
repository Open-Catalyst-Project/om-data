#!/bin/bash

num_lines=$(wc -l < elytes.csv)
num_lines=$((num_lines-1))

# Define the numbers as a space-separated list
# Loop through each number
for ((i = 1; i < num_lines; i++)); do
    rm $i -rf
    mkdir $i
    python generatesolvent_desmond.py $i
    echo "Waiting for solvent-out.cms ..."
    # Wait until the output file is generated
    while [ ! -f "./$i/solvent-out.cms" ]; do
        sleep 1  # Sleep for 1 second before checking again
    done
    
    cd $i
    $SCHRODINGER/utilities/multisim -o solvent_density.cms -mode umbrella solvent-out.cms -m solvent_multisim.msj -HOST localhost
    cd -

    echo "Waiting for solvent_density.cms ..."
    # Wait until the output file is generated
    while [ ! -f "./$i/solvent_density.cms" ]; do
        sleep 1  # Sleep for 1 second before checking again
    done
    
    #From here, check the whether the output is observed before moving on to the next stage
    $SCHRODINGER/run python3 computedensity.py $i
    python generatesystem_desmond.py $i
    echo "Waiting for elyte-out.cms ..."
    # Wait until the output file is generated
    while [ ! -f "./$i/elyte-out.cms" ]; do
        sleep 1  # Sleep for 1 second before checking again
    done
    
    cd $i
    $SCHRODINGER/utilities/multisim -o final_config.cms -mode umbrella elyte-out.cms -m elyte_multisim.msj -HOST localhost
    cd -
    echo "Waiting for final_config.cms ..."
    # Wait until the output file is generated
    while [ ! -f "./$i/final_config.cms" ]; do
        sleep 1  # Sleep for 1 second before checking again
    done
done
