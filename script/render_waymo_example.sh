#!/bin/bash
scenes=("007"  "015" "030" "031" "051" "130" "133" "159" "770") 
for scene in "${scenes[@]}"; do
    python render.py --config configs/configs_mvs/waymo_train_$scene.yaml mode evaluate
    python render.py --config configs/configs_mvs/waymo_train_$scene.yaml mode trajectory
done
