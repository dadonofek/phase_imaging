#!/bin/bash

for i in $(seq 1 10)
do
    dipole_len=$(echo "scale=1; $i / 10" | bc)
    python3 simulator.py --dipole_len $dipole_len

done


