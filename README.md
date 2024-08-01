# F1tenth-Offline-RL

F1tenth Autonomous Racing With Offline Reinforcement Learning Methods

For Return Conditioned Decision Tree Policy, --use_algorithm xgb.
For Decision Transformer Policy, --use_algorithm dt.

Training:
>> python3 train_racecar.py --use_algorithm xgb --training_dataset training_data/training_data_random_austria.csv --evaluate_in treitlstrasse_v2 --n_rounds 10000


>> python3 train_racecar.py --use_algorithm dt --training_dataset training_data/training_data_fixed_austria.csv --evaluate_in barcelona --n_rounds 100000



Simulation:

>> python3 simulate_racecar.py --use_algorithm xgb --model_path ./offline_models/xgb/trained_in_austria_10000_fixed_austria_20240108-185355 --n_episodes 20 --evaluate_in torino --render False


>> python3 simulate_racecar.py --use_algorithm dt --model_path ./offline_models/dt/trained_in_austria_100000_fixed_austria_20240119-154207 --n_episodes 20 --evaluate_in torino --render False


