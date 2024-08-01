import gymnasium
import racecar_gym.envs.gym_api
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium import spaces, Wrapper, Env, ObservationWrapper
from gymnasium.wrappers import TimeLimit
from gymnasium.spaces import Box
import numpy as np
import csv
import os


class Flatten(Wrapper):

    def __init__(self, env: gymnasium.Env):
        super(Flatten, self).__init__(env)
        self.action_space = spaces.flatten_space(env.action_space)
        self.action_space = Box(low=-1.0, high=1.0, shape=self.action_space.shape)
        ##obs to get only lidar
        self.observation_space = env.observation_space['lidar']

    def observation(self, observation):
        return observation['lidar']

    def action(self, action):
        return spaces.flatten(self.env.action_space, action)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = spaces.unflatten(self.env.action_space, action)
        obs, rewards, terminated, truncated, states = self.env.step(action)
        return obs['lidar'], rewards, terminated, truncated, states

    def reset(self, **kwargs):
        observation, states = self.env.reset(**kwargs)
        return observation['lidar'], states 







class Flatten_only_action(Wrapper):

    def __init__(self, env: gymnasium.Env):
        super(Flatten_only_action, self).__init__(env)
        self.action_space = spaces.flatten_space(env.action_space)
        self.action_space = Box(low=-1.0, high=1.0, shape=self.action_space.shape)


    def action(self, action):
        return spaces.flatten(self.env.action_space, action)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = spaces.unflatten(self.env.action_space, action)
        return self.env.step(action)



class OnlyCamera(Wrapper):

    def __init__(self, env: gymnasium.Env):
        super(OnlyCamera, self).__init__(env)
        self.action_space = spaces.flatten_space(env.action_space)
        self.action_space = Box(low=-1.0, high=1.0, shape=self.action_space.shape)
        ##obs to get only low_res_camera
        self.observation_space = env.observation_space['low_res_camera']

    def observation(self, observation):
        return observation['low_res_camera']

    def action(self, action):
        return spaces.flatten(self.env.action_space, action)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = spaces.unflatten(self.env.action_space, action)
        obs, rewards, terminated, truncated, states = self.env.step(action)
        return obs['low_res_camera'], rewards, terminated, truncated, states

    def reset(self, **kwargs):
        observation, states = self.env.reset(**kwargs)
        return observation['low_res_camera'], states



class RepeatAction(Wrapper):

    def __init__(self, env, n: int):
        self._repeat = n
        super().__init__(env)

    def step(self, action):
        obs, rewards, terminated, truncated, states = self.env.step(action)
        total_reward = rewards
        if not (terminated or truncated):
            for _ in range(self._repeat - 1):
                obs, rewards, terminated, truncated, states = self.env.step(action)
                total_reward += rewards
                if terminated or truncated:
                    break
        return obs, total_reward, terminated, truncated, states