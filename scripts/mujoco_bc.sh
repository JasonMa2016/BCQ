#!/usr/bin/env bash

ENV="Walker2d-v2"
TRAJ=1

for seed in 0 1 2 3 4
do
    python3 train_bc.py --env_name $ENV --num_trajs $TRAJ --seed $seed --good
    python3 train_bc.py --env_name $ENV --num_trajs $TRAJ --seed $seed
done


