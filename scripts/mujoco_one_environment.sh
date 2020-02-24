#!/usr/bin/env bash

ENV="HalfCheetah-v2"

python3 train_bc.py --env_name $ENV --num_trajs 5 --seed 0 --good --ensemble
for seed in 0 1 2 3 4
do
    python3 train_bcq_traj.py --env_name $env --num_trajs 5 --seed $seed --good
    python3 train_drbcq_traj.py --env_name $env --num_trajs 5 --seed $seed --good
done
