#!/usr/bin/env bash

for env in Hopper-v2
do
    python3 train_bc.py --env_name $env --num_trajs 5 --seed 0
    python3 train_bc.py --env_name $env --num_trajs 5 --seed 0 --good
    python3 train_drbcq_traj.py --env_name $env --num_trajs 5 --seed 0
    python3 train_drbcq_traj.py --env_name $env --num_trajs 5 --seed 0 --good
done