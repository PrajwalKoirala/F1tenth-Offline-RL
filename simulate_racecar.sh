#!/bin/bash


ALGORITHM="xgb"
MODEL_PATH="./offline_models/xgb/trained_in_AuBaTr_20000_mixed_20240124-142540"
N_EPISODES=20
RENDER="False"

LOCATIONS=("austria" "barcelona" "berlin" "columbia" "torino" "treitlstrasse_v2")


for location in "${LOCATIONS[@]}"; do
    echo "---------------------------- $location --------------------------------"
    echo "Running simulation for location: $location"
    python3 simulate_racecar.py --use_algorithm $ALGORITHM --model_path $MODEL_PATH --n_episodes $N_EPISODES --evaluate_in $location --render $RENDER
done