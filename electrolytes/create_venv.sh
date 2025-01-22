#!/bin/bash

$SCHRODINGER/run schrodinger_virtualenv.py $1
source $1/bin/activate
python3 -m pip install pulp
deactivate
