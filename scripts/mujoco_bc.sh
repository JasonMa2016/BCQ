#!/usr/bin/env bash

ENV="Hopper-v2"

python3 train_bc.py --env_name $ENV --num_trajs 5 --seed 0 --good --ensemble
python3 train_bc.py --env_name $ENV --num_trajs 5 --seed 0 --ensemble

