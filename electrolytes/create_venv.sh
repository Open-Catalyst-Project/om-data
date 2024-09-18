#!/bin/bash

$SCHRODINGER/run schrodinger_virtualenv.py /checkpoint/levineds/elytes
source /checkpoint/levineds/elytes/bin/activate
python3 -m pip install pulp
deactivate
