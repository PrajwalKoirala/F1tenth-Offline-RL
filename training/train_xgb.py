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

def discount_cumsum(x, gamma=1):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(len(x) - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

def sum_of_rewards(x):
    total_reward = np.zeros_like(x)
    total_reward += np.sum(x)
    return total_reward

def sum_of_next_n_rewards(x, window_size=10):
    total_reward = np.zeros_like(x)
    for i in range(len(x)):
        end_index = min(i + window_size, len(x))
        num_elements = end_index - i
        total_reward[i] = np.sum(x[i:end_index]) * (window_size / num_elements)
    total_reward[-2:] = total_reward[-3]
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


def read_data(dataset_path='/home/prajwal/workfolder/racecar_gym/map_based_control/OfflineData/Austria/austria_data.csv'):
    with open(dataset_path) as f:
        temp = f.readlines()
    temp = [x.split(',') for x in temp]
    data = [[float(y) for y in x] for x in temp]
    data = np.array(data)
    print('Data shape:', data.shape)
    return data

def preprocess_data(data, n_points=20, n_repeat=3):
    dones = data[:, -2] + data[:, -1]
    terminal_ind = np.where(dones)[0]
    if not dones[-1]:
        terminal_ind = np.concatenate((terminal_ind, np.array([len(dones)-1])))
    start_ind = np.concatenate((np.array([0]), terminal_ind+1))
    rtg = []
    sor = []
    timesteps = []
    gamma = 1
    new_obs = []
    poses = []
    velocities = []
    for i in range(len(start_ind)-1):
        rewards = data[start_ind[i]:start_ind[i+1], -3]
        rtg.extend(discount_cumsum(rewards, gamma))
        timesteps.extend(np.arange(len(rewards)))
        lidar_data = data[start_ind[i]:start_ind[i+1], 6:-5]
        new_obs.append(augment_obs(lidar_data, n=n_repeat, n_points=n_points))
        poses.append(data[start_ind[i]:start_ind[i+1], :3])
        velocities.append(augment_obs(data[start_ind[i]:start_ind[i+1], 3:6], n=n_repeat))

    rtg = np.array(rtg).reshape(-1, 1)
    timesteps = np.array(timesteps).reshape(-1, 1)
    obs = np.concatenate(new_obs, axis=0)
    poses = np.concatenate(poses, axis=0)
    velocities = np.concatenate(velocities, axis=0)
    X = np.concatenate((obs, velocities, rtg, timesteps), axis=1)
    y = data[:, -5:-3]
    y = np.clip(y, -1, 1)
    print('X-shape:', X.shape, ',and y-shape:', y.shape)
    return X, y

def train_xgb(training_dataset, output_path, n_rounds=10_000):
    data = read_data(dataset_path=training_dataset)
    X, y = preprocess_data(data, n_points=20, n_repeat=3)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', 
                                 n_estimators=n_rounds, max_depth=10,
                                 random_state=123)
    starting_time = time.time()
    xgb_model.fit(X, y, verbose=True)
    print('Training time:', time.time() - starting_time)
    print('Model trained.')
    print('Training MSE:', np.mean((xgb_model.predict(X) - y)**2))
    xgb_model.save_model(output_path)
    print('Model saved at:', output_path)



    