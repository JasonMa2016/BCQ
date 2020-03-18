#!/usr/bin/env bash

TRAJ=1
ENV='Walker2d-v2'
do
    for seed in 1 2 3 4
    do
#        python3 train_bc.py --env_name $env --num_trajs $TRAJ --seed $seed --type "good"
#        python3 train_bc.py --env_name $env --num_trajs $TRAJ --seed $seed --type "mixed"
        python3 train_bc.py --env_name $ENV --num_trajs $TRAJ --seed $seed --type "imperfect"
    done
done


