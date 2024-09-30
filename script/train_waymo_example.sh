#!/bin/bash
scenes=("007")
for scene in "${scenes[@]}"; do
    python train.py --config configs/example/waymo_train_$scene.yaml
done
