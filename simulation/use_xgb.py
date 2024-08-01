import random
import numpy as np
import xgboost as xgb
import time
import gymnasium
import racecar_gym.envs.gym_api
from gymnasium.spaces import Box
import numpy as np
from custom_wrappers import *
import pybullet as p


def sum_of_rewards(x):
    total_reward = np.zeros_like(x)
    total_reward += np.sum(x)
    return total_reward


def extract_sparse_points(data, m):
    n, q = data.shape 
    indices = np.linspace(0, q-1, m, dtype=int)  
    extracted_data = data[:, indices]  
    return extracted_data



def augment_obs(initial_obs, n=2, n_points=20):
    rows, cols = initial_obs.shape
    augmented_obs = []
    for i in range(rows):
        obs = []
        for j in range(n, 0, -1):
            if i - j + 1 >= 0:
                obs.extend(initial_obs[i - j + 1])
            else:
                obs.extend(initial_obs[0])
        augmented_obs.append(obs)
    return np.array(augmented_obs)



def use_xgb(env_name='austria', n_episodes=10, n_repeat=3, n_points=20, model_path='offline_models/xgb/trained_in_austria', render=False):
    n_repeat = n_repeat
    env_name= env_name.capitalize()
    EPISODES = n_episodes
    render_mode = 'human' if render else 'None'
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(model_path)
    env = gymnasium.make(
        id=f'SingleAgent{env_name}-v0',
        render_mode=render_mode,
        scenario=f'scenarios/{env_name.lower()}.yml',
    )

    env = RepeatAction(env, 4)
    env = Flatten_only_action(env)

    reset_options = dict(mode='grid')
    obs, info = env.reset(options=reset_options)
    p.resetDebugVisualizerCamera(cameraDistance=43.6, cameraYaw=78, cameraPitch=-55, cameraTargetPosition=[-4.07,-8.7,-22.28])

    done = False
    all_episode_returns = []
    inference_times = []


    for episode in range(EPISODES):
        print(f"Episode {episode+1} started")
        done = False
        reset_options = dict(mode='grid')
        obs, info = env.reset(options=reset_options)
        lidar = obs['lidar']
        this_obs = extract_sparse_points(np.array(lidar).reshape(1, -1), n_points)
        this_obs = np.repeat(this_obs, n_repeat, axis=1)
        this_vel = np.concatenate((obs['velocity'][0:2], [obs['velocity'][5]])).reshape(1, -1)
        this_vel = np.repeat(this_vel, n_repeat, axis=1)
        target_rtg = 99.5 # hyper parameter
        totalreward = 0
        episode_length = 0
        while not done:
            lidar = extract_sparse_points(np.array(obs['lidar']).reshape(1, -1), n_points)
            this_obs = np.concatenate((this_obs[:, lidar.shape[1]:], lidar), axis=1) 
            vel = np.concatenate((obs['velocity'][0:2], [obs['velocity'][5]])).reshape(1, -1)
            this_vel = np.concatenate((this_vel[:, vel.shape[1]:], vel), axis=1)
            start_time = time.time()
            action = xgb_model.predict(np.concatenate(( this_obs[0], this_vel[0], [target_rtg, episode_length] )).reshape(1, -1))[0]
            inference_times.append(time.time() - start_time)
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            totalreward += rewards
            episode_length += 1
            target_rtg -= rewards
        print(f"Episode {episode+1} finished after {episode_length} timesteps, totalreward: {totalreward}")
        print("-----------------------------------------")
        all_episode_returns.append(totalreward)
        
    mean_return = np.mean(all_episode_returns)
    std_return = np.std(all_episode_returns)
    print(f"Average return: {mean_return}, std return: {std_return}")
    print(f"Max return: {np.max(all_episode_returns)}")
    print(f"Min return: {np.min(all_episode_returns)}")
    print(f"Average inference time: {np.mean(inference_times)}, Std inference time: {np.std(inference_times)}")
    # temp = input("\nPress enter to exit")
    env.close()
    p.disconnect()
    return mean_return, std_return