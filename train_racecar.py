import argparse
import os
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train racecar agent.')

    parser.add_argument('--use_algorithm', type=str, choices=['xgb', 'dt'], 
                        help='Which algorithm do you want to use?', default='xgb')
    parser.add_argument('--n_rounds', type=int, help='Number of epochs or boosting rounds to run.')
    parser.add_argument('--training_dataset', type=str,
                        help='Path of the training dataset.', default='training_data/training_data_fixed_austria.csv')
    parser.add_argument('--training_track', type=str, choices=['austria', 'barcelona', 'treitlstrasse'],
                        help='Which track was used to collect the training data?')
    parser.add_argument('--output_path', type=str,
                        help='Path to save the model.')
    parser.add_argument('--evaluate', type=bool, help='Evaluate the model?', default=True)

    all_scenarios = [x.split('.')[0] for x in os.listdir('scenarios')]
    parser.add_argument('--evaluate_in', type=str, choices=all_scenarios,
                        help='Which track do you want to evaluate in?')
    args = parser.parse_args()
    if args.training_track == None:
        args.training_track = args.training_dataset.split('.')[1].split('_')[-1]
    if args.evaluate_in == None:
        args.evaluate_in = args.training_track
    if args.n_rounds == None:
        if args.use_algorithm == 'xgb':
            args.n_rounds = 10_000
        elif args.use_algorithm == 'dt':
            args.n_rounds = 50_000
    if args.output_path == None:
        args.output_path = (
            f"offline_models/{args.use_algorithm}/"
            f"trained_in_{args.training_track}_"
            f"{args.n_rounds}_"
            f"{'_'.join(args.training_dataset.split('.')[1].split('_')[-2:])}_"
            f"{time.strftime('%Y%m%d-%H%M%S')}"
        )
    print(args.output_path)
    if args.evaluate and args.evaluate_in == None:
        args.evaluate_in = args.training_track

    return args

def main():
    args = parse_arguments()
    alg = args.use_algorithm
    n_rounds = args.n_rounds
    training_dataset = args.training_dataset
    do_evaluate = args.evaluate

    if alg == 'xgb':
        import training.train_xgb as train_xgb
        train_xgb.train_xgb(n_rounds=n_rounds, training_dataset=training_dataset, 
                            output_path=args.output_path)
        if do_evaluate:
            import simulation.use_xgb as use_xgb
            use_xgb.use_xgb(env_name=args.evaluate_in, n_episodes=10, n_repeat=3, 
                            n_points=20, model_path=args.output_path, render=False)
    elif alg == 'dt':
        import training.train_dt as train_dt
        train_dt.train_dt(n_epochs=n_rounds, training_dataset=training_dataset, 
                          output_path=args.output_path)
        if do_evaluate:
            import simulation.use_dt as use_dt
            use_dt.use_dt(env_name=args.evaluate_in, n_episodes=10, model_path=args.output_path, 
                          render=False)
        

    

if __name__ == '__main__':
    main()
