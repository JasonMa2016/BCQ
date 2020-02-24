#!/usr/bin/env bash

for env in Reacher-v2 HalfCheetah-v2 Walker2d-v2 Ant-v2 Humanoid-v2
do
    for seed in 0
    do
        python3 generate_trajectories.py --env_name $env --seed $seed
    done
done
