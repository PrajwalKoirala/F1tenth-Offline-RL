import os
import random
from dataclasses import dataclass
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments
import json
import time
import yaml


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

def preprocess_data(data):
    dones = data[:, -2] + data[:, -1]
    terminal_ind = np.where(dones)[0]
    if not dones[-1]:
        terminal_ind = np.concatenate((terminal_ind, np.array([len(dones)-1])))
    start_ind = np.concatenate((np.array([0]), terminal_ind+1))
    gamma = 1
    rewards = []
    new_obs = []
    poses = []
    velocities = []
    actions = []
    dones = []
    for i in range(len(start_ind)-1):
        this_rewards = data[start_ind[i]:start_ind[i+1], -3]
        rewards.append(this_rewards)
        lidar_data = data[start_ind[i]:start_ind[i+1], 6:-5]
        new_obs.append(lidar_data)
        poses.append(data[start_ind[i]:start_ind[i+1], :3])
        velocities.append(data[start_ind[i]:start_ind[i+1], 3:6])
        actions.append(data[start_ind[i]:start_ind[i+1], -5:-3])
        dones.append(data[start_ind[i]:start_ind[i+1], -2] + data[start_ind[i]:start_ind[i+1], -1])

    dataset = []

    for vels, obs, acts, rews, dones in zip(velocities, new_obs, actions, rewards, dones):
        observ = np.concatenate((vels, obs), axis=1)
        episode = {
            'observations': observ,
            'actions': acts,
            'rewards': rews,
            'dones': dones
        }
        dataset.append(episode)
    return dataset


@dataclass
class DecisionTransformerGymDataCollator:
    return_tensors: str = "pt"
    max_len: int = 20 #subsets of the episode we use for training
    state_dim: int = 1080  # size of state space
    act_dim: int = 2  # size of action space
    max_ep_len: int = 3500 # max episode length in the dataset
    scale: float = 100.0  # normalization of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0 # to store the number of trajectories in the dataset

    def __init__(self, dataset) -> None:
        self.act_dim = len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["observations"][0])
        self.dataset = dataset
        # calculate dataset stats for normalization of states
        states = []
        traj_lens = []
        obs_data = [dataset[i]['observations'] for i in range(len(dataset))]
        for obs in obs_data:
            states.extend(obs)
            traj_lens.append(len(obs))
        self.n_traj = len(traj_lens)
        states = np.vstack(states)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        traj_lens = np.array(traj_lens)
        self.p_sample = traj_lens / sum(traj_lens)

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)
        # this is a bit of a hack to be able to sample of a non-uniform distribution
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )
        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

        for ind in batch_inds:
            # for feature in features:
            feature = self.dataset[int(ind)]
            si = random.randint(0, len(feature["rewards"]) - 1)

            # get sequences from dataset
            s.append(np.array(feature["observations"][si : si + self.max_len]).reshape(1, -1, self.state_dim))
            a.append(np.array(feature["actions"][si : si + self.max_len]).reshape(1, -1, self.act_dim))
            r.append(np.array(feature["rewards"][si : si + self.max_len]).reshape(1, -1, 1))

            d.append(np.array(feature["dones"][si : si + self.max_len]).reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature["rewards"][si:]), gamma=1.0)[
                    : s[-1].shape[1]   # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }


class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = torch.mean((action_preds - action_targets) ** 2)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)


def train_dt(training_dataset, output_path, n_epochs=100_000):
    data = read_data(dataset_path=training_dataset)
    data = preprocess_data(data)
    collator = DecisionTransformerGymDataCollator(data)
    config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim, max_length=collator.max_len)
    model = TrainableDT(config)

    training_args = TrainingArguments(
        output_dir="output/",
        remove_unused_columns=False,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=64,
        learning_rate=5e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        optim="adamw_torch",
        max_grad_norm=0.25,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
        data_collator=collator,
    )
    starting_time = time.time()
    trainer.train()
    print('Training time:', time.time() - starting_time)
    print('Model trained.')
    trainer.save_model(output_path)
    print('Model saved at:', output_path)
    collator_params = {
        'state_dim': collator.state_dim,
        'act_dim': collator.act_dim,
        'state_mean': collator.state_mean.tolist() if isinstance(collator.state_mean, np.ndarray) else collator.state_mean,
        'state_std': collator.state_std.tolist() if isinstance(collator.state_std, np.ndarray) else collator.state_std,
        'scale': collator.scale.tolist() if isinstance(collator.scale, np.ndarray) else collator.scale,
        'max_ep_len': collator.max_ep_len.tolist() if isinstance(collator.max_ep_len, np.ndarray) else collator.max_ep_len,
        'max_len': collator.max_len.tolist() if isinstance(collator.max_len, np.ndarray) else collator.max_len,
        'n_traj': collator.n_traj.tolist() if isinstance(collator.n_traj, np.ndarray) else collator.n_traj,
        'p_sample': collator.p_sample.tolist() if isinstance(collator.p_sample, np.ndarray) else collator.p_sample
    }
    with open(f'{output_path}/collator_config.yml', 'w') as outfile:
        yaml.dump(collator_params, outfile, default_flow_style=False)
    for key, value in collator_params.items():
        print(f"{key}: {value}")



    