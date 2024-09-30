#!/bin/bash
scenes=("007" "015" "030" "051" "130" "133" "159" "770") 
for scene in "${scenes[@]}"; do
    python train.py --config configs/experiments_waymo/waymo_val_$scene.yaml
done
