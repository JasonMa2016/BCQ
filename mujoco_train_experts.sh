#!/usr/bin/env bash

for env in Hopper-v2
do
    python3 train_expert.py --env-name $env --start-timesteps 1000
done
for env in Reacher-v2 HalfCheetah-v2 Walker2d-v2 Ant-v2 Humanoid-v2
do
    python3 train_expert.py --env-name $env
done