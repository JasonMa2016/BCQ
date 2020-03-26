#!/usr/bin/env bash

ENV='Hopper-v2'
for traj in 1 3 5
do
for seed in 0 1 2 3 4
do
    python3 train_drbcq_traj.py --env_name $ENV --num_trajs $traj --seed $seed --type "good"
    python3 train_drbcq_traj.py --env_name $ENV --num_trajs $traj --seed $seed --type "mixed"
done
done

