import argparse
import gym
import numpy as np
import os
import random
import torch
import logging
import os

from utils import mkdir
from agent import DQNAgent



def main():
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--env', type=str, default='MsPacman-v0') # 'Breakout-v0'
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps', type=float, default=1.0)
    parser.add_argument('--exploration_decay_speed', type=int, default=1000000)
    parser.add_argument('--eps_min', type=float, default=0.1)
    parser.add_argument('--log_size', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--buffer_init_size', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sync_period', type=int, default=10000)
    parser.add_argument('--learn_freq', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp-dir', type=str, default='exp')
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() \
        and args.device.startswith('cuda') else 'cpu')

    work_dir = mkdir(args.exp_dir, args.env) # save models

    # logging infos
    logging.basicConfig(filename=args.env+'.log', filemode='w', level=logging.INFO)

    env = gym.make(args.env)

    # set seed
    env.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    agent = DQNAgent(env, args, work_dir)
    agent.run()



if __name__ == '__main__':
    main()
