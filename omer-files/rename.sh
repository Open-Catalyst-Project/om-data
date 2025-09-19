#!/bin/bash

for dir in */; do
dir=${dir%/}
    label="${dir}"
    src="$dir"/frame*.pdb
    dest=("$dir/${label}_Hterm.pdb")
    if [ -f $src ]; then
        mv $src $dest
        echo "Renamed $src to $dest"
    else
        echo "No clean.pdb in $dir"
    fi
done