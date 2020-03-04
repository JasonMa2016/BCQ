#!/usr/bin/env bash

ENV="Walker2d-v2"

for seed in 0 1 2 3 4
do
    python3 train_bcq_traj.py --env_name $ENV --num_trajs 5 --seed $seed
    python3 train_drbcq_traj.py --env_name $ENV --num_trajs 5 --seed $seed --random
done
