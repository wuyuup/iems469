import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Policy
import gym
import utils
from storage import RolloutStorage
from collections import deque
import time
from itertools import count
import os
import logging

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(agent, shared_model, args, device, idx, counter, lock):
    torch.manual_seed(args.seed + idx)
    env = gym.make(args.env_name)
    env.seed(args.seed + idx)
    obs = torch.from_numpy(utils.preprocess(env.reset())).float().unsqueeze(0).unsqueeze(0)
    # local model
    local_ac = Policy(obs.shape)
    local_ac.to(device)
    # rollout buffer
    rollouts = RolloutStorage(args.num_steps, obs.shape, env.action_space, device)
    # rollouts = RolloutStorage(args.num_steps, obs.shape, env.action_space, 'cpu')

    rollouts.obs[0] = torch.Tensor(obs)

    episode_rewards = deque(maxlen = 10)

    epi_rew = 0

    optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr, eps=args.eps, alpha=args.alpha)

    for j in count(1):
        local_ac.load_state_dict(shared_model.state_dict())
        step = 0
        done = False
        for step in range(args.num_steps):
            # action: from last buffer
            with torch.no_grad():
                value, action, action_log_prob = local_ac(rollouts.obs[step])

            # action +2: left and right
            obs, reward, done, info = env.step(action+2)
            obs = torch.from_numpy(utils.preprocess(obs)).float().unsqueeze(0).unsqueeze(0)
            epi_rew += reward

            if done:
                episode_rewards.append(epi_rew)
                epi_rew = 0
                obs = torch.from_numpy(utils.preprocess(env.reset())).float().unsqueeze(0).unsqueeze(0)

            masks = torch.FloatTensor([[0.0] if done else [1.0]])
            rollouts.insert(obs, action[0], action_log_prob[0], value[0], reward, masks[0])

            with lock:
                counter.value += 1

        # PV
        with torch.no_grad():
            next_value = local_ac.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args)

        total_loss = agent.update(rollouts)

        optimizer.zero_grad()
        total_loss.backward()
        # min(max, gradient)
        nn.utils.clip_grad_norm_(local_ac.parameters(), args.max_grad_norm)
        ensure_shared_grads(local_ac, shared_model)
        optimizer.step()

        # need to keep mask and obs..
        rollouts.after_update()

        # save model: ok to save local model (load from shared model)
        if (j % args.save_interval == 0) and args.save_dir != "":
            cur_path = os.path.join(args.save_dir)
            try:
                os.makedirs(cur_path)
            except OSError:
                pass
            torch.save([local_ac], os.path.join(cur_path, args.env_name + ".pt"))

        # logging
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_steps
            output_str = "Updates {}, num timesteps {} \n Last {} training episodes: mean reward {:.1f}".format(
                j, total_num_steps, len(episode_rewards), np.mean(episode_rewards))

            print(output_str)
            logging.info(output_str)


