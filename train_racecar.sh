#!/bin/bash

ALGORITHM="xgb"
TRAINING_DATASET="./training_data/training_data_fixed_austria.csv"
EVALUATE_IN="austria"
N_ROUNDS_XGB=10000
N_ROUNDS_DT=100000

echo "---------------------------- Training Racecar --------------------------------"

if [ "$ALGORITHM" == "xgb" ]; then
    N_ROUNDS=$N_ROUNDS_XGB
elif [ "$ALGORITHM" == "dt" ]; then
    N_ROUNDS=$N_ROUNDS_DT
else
    echo "Unknown algorithm: $ALGORITHM"
    exit 1
fi

echo "Training racecar with algorithm: $ALGORITHM"
python3 train_racecar.py --use_algorithm $ALGORITHM --training_dataset $TRAINING_DATASET --evaluate_in $EVALUATE_IN --n_rounds $N_ROUNDS