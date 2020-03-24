#!/usr/bin/env bash

TRAJ=5
ENV='Walker2d-v2'

for seed in 0 1 2 3 4
    do
        python3 train_drbcq_traj.py --env_name $ENV --num_trajs $TRAJ --seed $seed --type "good"
        python3 train_drbcq_traj.py --env_name $ENV --num_trajs $TRAJ --seed $seed --type "mixed"
done

