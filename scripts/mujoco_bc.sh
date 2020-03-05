#!/usr/bin/env bash

for env in Hopper-v2 Walker2d-v2 Humanoid-v2
do
    python3 train_bc.py --env_name $ENV --num_trajs 5 --seed 0 --good --ensemble
    python3 train_bc.py --env_name $ENV --num_trajs 5 --seed 0 --ensemble
done

