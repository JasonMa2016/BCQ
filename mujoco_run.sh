#!/usr/bin/env bash

for env in Reacher-v2 Hopper-v2 HalfCheetah-v2 Walker2d-v2 Ant-v2 Humanoid-v2
do
    python3 generate_trajectories.py --env_name $env
    python3 train_bc.py --env_name $env --num_trajs 5
    python3 train_drbcq_traj.py --env_name $env --num_trajs 5
done