import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Simulate the F1tenth Racecar.')

    parser.add_argument('--use_algorithm', type=str, choices=['xgb', 'dt'], 
                        help='Which algorithm do you want to use?', default='xgb')
    parser.add_argument('--n_episodes', type=int, help='Number of episodes to run.', default=10)
    parser.add_argument('--trained_in', type=str, choices=['austria'], 
                        help='Which track was the model trained in?', default='austria')
    parser.add_argument('--vehicle_init', type=str, choices=['random', 'fixed'],
                        help='How do you want to initialize the vehicle?', default='fixed')
    parser.add_argument('--model_path', type=str, help='Path to the model to use.')
    
    all_scenarios = [x.split('.')[0] for x in os.listdir('scenarios')]
    parser.add_argument('--evaluate_in', type=str, choices=all_scenarios,
                        help='Which track do you want to evaluate in?', default='austria')
    parser.add_argument('--render', type=bool, help='Render the simulation?', default=False)
    
    args = parser.parse_args()
    if args.vehicle_init == 'random':
        args.vehicle_init = 'fixed'
        print('Not implemented yet. Initializing vehicle at origin.')
    if args.model_path is None:
        args.model_path = f'offline_models/{args.use_algorithm}/trained_in_{args.trained_in}'
    
    return args

def main():
    args = parse_arguments()

    alg = args.use_algorithm
    n_episodes = args.n_episodes
    model_path = args.model_path

    
    if alg == 'xgb':
        import simulation.use_xgb as use_xgb
        if model_path is None:
            model_path = f'offline_models/xgb/trained_in_{args.trained_in}'
        mean_return, std_return  = use_xgb.use_xgb(env_name=args.evaluate_in, 
                                                   n_episodes=n_episodes, n_repeat=3, 
                                                   n_points=20, model_path=model_path, 
                                                   render=args.render)
    elif alg == 'dt':
        import simulation.use_dt as use_dt
        if model_path is None:
            model_path = f'offline_models/dt/trained_in_{args.trained_in}'
        mean_return, std_return  = use_dt.use_dt(env_name=args.evaluate_in, 
                                                 n_episodes=n_episodes, model_path=model_path, 
                                                 render=args.render)

    

if __name__ == '__main__':
    main()
