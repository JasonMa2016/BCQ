#!/usr/bin/env bash

for env in Hopper-v2
do
    python3 train_bc.py --env_name $env --num_trajs 5 --seed 0 --ensemble
#    python3 train_bc.py --env_name $env --num_trajs 5 --seed 0 --good --ensemble
done
for env in Hopper-v2
do
    for seed in 0 1 2
    do
        python3 train_drbcq_traj.py --env_name $env --num_trajs 5 --seed $seed
#        python3 train_drbcq_traj.py --env_name $env --num_trajs 5 --seed $seed --good
#        python3 train_bcq_traj.py --env_name $env --num_trajs 5 --seed $seed
#        python3 train_bcq_traj.py --env_name $env --num_trajs 5 --seed $seed --good
    done
done
