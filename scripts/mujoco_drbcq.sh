#!/usr/bin/env bash

TRAJ=5

for env in Walker2d-v2
do
    for seed in 0 1 2 3 4
    do
        python3 train_drbcq_traj.py --env_name $env --num_trajs $TRAJ --seed $seed --good
        python3 train_drbcq_traj.py --env_name $env --num_trajs $TRAJ --seed $seed
    done
done
