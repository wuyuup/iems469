import numpy as np
import matplotlib.pyplot as plt
import gym
import sys
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import logging

seed = 42
gamma = 0.95
log_interval = 1
env = gym.make('CartPole-v0')
env.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.linear_layer1 = nn.Linear(4, 16)
        self.linear_layer2 = nn.Linear(16, 2)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.linear_layer1(x)
        x = F.relu(x)
        output = self.linear_layer2(x)
        return F.softmax(output, dim=1)

policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample().to(device)
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def optimize():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns).to(device)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * (R - returns.mean() ))
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum().to(device)
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    logging.basicConfig(filename='cartpole.log', filemode='w', level=logging.INFO)
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        done = False
        while not done:
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward


        running_reward = 0.1 * ep_reward + (1 - 0.1) * running_reward
        optimize()
        if i_episode % log_interval == 0:
            logging.info('Episode {}\tLast reward: {:.2f}\tLast 10 Average reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
            print(ep_reward, running_reward)

main()
