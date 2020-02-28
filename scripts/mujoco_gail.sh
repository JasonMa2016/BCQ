#!/usr/bin/env bash

ENV="HalfCheetah-v2"

for num_traj in 1 3 5
do
    for seed in 0
    do
        python3 train_gail_ppo_expert.py --env_name $ENV --num_trajs $num_traj --seed $seed
        python3 train_gail_ppo_expert.py --env_name $ENV --num_trajs $num_traj --seed $seed --good
    done
done
