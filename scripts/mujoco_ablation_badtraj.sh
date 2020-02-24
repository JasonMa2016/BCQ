#!/usr/bin/env bash

for env in Hopper-v2
do
    for seed in 0 1 2 3 4
    do
        python3 train_drbcq_traj.py --env_name $env --num_bad_trajs 10 --seed $seed
        python3 train_drbcq_traj.py --env_name $env --num_bad_trajs 15 --seed $seed
    done
done
