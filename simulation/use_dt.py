import gymnasium
import racecar_gym.envs.gym_api
from gymnasium import spaces, Wrapper, Env, ObservationWrapper
from gymnasium.spaces import Box
import numpy as np
import os
from custom_wrappers import *
import random
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments
import json
import yaml
import time
import pybullet as p



def extract_sparse_points(data, m):
    n, q = data.shape 
    indices = np.linspace(0, q-1, m, dtype=int)  
    extracted_data = data[:, indices]  
    return extracted_data


def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    padding = model.config.max_length - states.shape[1]
    # pad all tokens to sequence length
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    state_preds, action_preds, return_preds = model.forward(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )
    return action_preds[0, -1]




def use_dt(env_name='austria', n_episodes=10, model_path='offline_models/xgb/trained_in_austria', render=False):
    env_name= env_name.capitalize()
    EPISODES = n_episodes
    render_mode = 'human' if render else 'None'
    env = gymnasium.make(
        id=f'SingleAgent{env_name}-v0',
        render_mode=render_mode,
        scenario=f'scenarios/{env_name.lower()}.yml',
    )
    env = Flatten(env)
    env = RepeatAction(env, 4)

    with open(f"{model_path}/collator_config.yml", "r") as f:
        collator_config = yaml.safe_load(f)


    device = "cpu"
    # "/home/prajwal/workfolder/racecar_gym/DT_based_control/dt_model_racecar_ctx_20"
    model = DecisionTransformerModel.from_pretrained(model_path)
    model = model.to(device=device)
    scale = 100
    TARGET_RETURN = 100.5/scale

    state_dim = collator_config["state_dim"]
    act_dim = collator_config["act_dim"]
    state_mean = torch.from_numpy(np.array(collator_config["state_mean"]).astype(np.float32)).to(device=device)
    state_std = torch.from_numpy(np.array(collator_config["state_std"]).astype(np.float32)).to(device=device)

    reset_options = dict(mode='grid')
    obs, info = env.reset(options=reset_options)
    p.resetDebugVisualizerCamera(cameraDistance=43.6, cameraYaw=78, cameraPitch=-55, cameraTargetPosition=[-4.07,-8.7,-22.28])

    done = False

    all_episode_returns = []
    inference_time = []


    for episode in range(EPISODES):
        print(f"Episode {episode+1} started")
        done = False
        reset_options = dict(mode='grid')
        obs, info = env.reset(options=reset_options)
        target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
        vel = np.concatenate((info['velocity'][0:2], [info['velocity'][5]])).reshape(1, -1)
        obs  = extract_sparse_points(obs.reshape(1, -1),20)
        obs = np.concatenate([vel, obs], axis=1)
        states = torch.from_numpy(obs).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        episode_length = 0
        episode_return = 0
        while True:
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            start_time = time.time()
            action = get_action(
                model,
                (states - state_mean) / state_std,
                actions,
                rewards,
                target_return,
                timesteps,
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()
            inference_time.append(time.time() - start_time)
            obs, reward, terminated, truncated, states_info = env.step(action)
            done = terminated or truncated

            obs = extract_sparse_points(obs.reshape(1, -1),20)
            vel = np.concatenate((states_info['velocity'][0:2], [states_info['velocity'][5]])).reshape(1, -1)
            obs = np.concatenate([vel, obs], axis=1)
            cur_state = torch.from_numpy(obs).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            pred_return = target_return[0, -1] - (reward / scale)
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (episode_length + 1)], dim=1)
            episode_return += reward
            episode_length += 1
            states = states[-model.config.max_length :,:]
            if done:
                print(f"Episode {1} finished after {episode_length} timesteps, totalreward: {episode_return}")
                print("-----------------------------------------")
                all_episode_returns.append(episode_return)
                break
        
    mean_return = np.mean(all_episode_returns)
    std_return = np.std(all_episode_returns)
    print(f"Average return: {mean_return}, std return: {std_return}")
    print(f"Max return: {np.max(all_episode_returns)}")
    print(f"Min return: {np.min(all_episode_returns)}")
    print(f"Average inference time: {np.mean(inference_time)}, Std inference time: {np.std(inference_time)}")
    # temp = input("\nPress enter to exit")
    env.close()
    p.disconnect()
    return mean_return, std_return