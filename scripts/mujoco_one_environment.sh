#!/usr/bin/env bash

ENV="Walker2d-v2"

#python3 train_bc.py --env_name $ENV --num_trajs 5 --seed 0 --good --ensemble
#python3 train_bc.py --env_name $ENV --num_trajs 5 --seed 0 --ensemble

for seed in 3 4  
do
    python3 train_bcq_traj.py --env_name $ENV --num_trajs 5 --seed $seed --good
    python3 train_drbcq_traj.py --env_name $ENV --num_trajs 5 --seed $seed --good
    python3 train_drbcq_traj.py --env_name $ENV --num_trajs 5 --seed $seed
    python3 train_drbcq_traj.py --env_name $ENV --num_trajs 5 --seed $seed --good --random
done
