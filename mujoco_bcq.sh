#!/usr/bin/env bash

for env in Hopper-v2
do
    python3 train_bcq_traj.py --env_name $env --num_trajs 5 --seed 0
    python3 train_bcq_traj.py --env_name $env --num_trajs 5 --seed 0 --good
done