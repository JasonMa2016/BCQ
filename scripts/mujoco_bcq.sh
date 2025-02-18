#!/usr/bin/env bash

for env in Reacher-v2 Hopper-v2 HalfCheetah-v2 Walker2d-v2 Ant-v2 Humanoid-v2
do
    for seed in 0 1 2 3 4
    do
        python3 train_bcq_traj.py --env_name $env --num_trajs 5 --seed $seed
        python3 train_bcq_traj.py --env_name $env --num_trajs 5 --seed $seed --good
    done
done
