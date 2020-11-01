import copy
import glob
import os
import time
from collections import deque
from itertools import count

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
from arguments import get_args
from model import Policy
from storage import RolloutStorage
from a2c import A2C
import logging
import torch.multiprocessing as mp
from train import train


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(filename=args.log_dir, filemode='w', level=logging.INFO)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    device = torch.device('cuda' if args.cuda else 'cpu')
    # device = 'cpu'
    env = gym.make(args.env_name)
    env.seed(args.seed)

    # reset env and preprocess to obtain the shape of the input (feeded into nn)
    obs = torch.from_numpy(utils.preprocess(env.reset())).float().unsqueeze(0).unsqueeze(0)

    shared_ac = Policy(obs.shape)
    shared_ac.to(device)
    shared_ac.share_memory()


    agent = A2C(shared_ac, args)

    if args.cuda:
        torch.multiprocessing.set_start_method('spawn')

    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    for idx in range(0, args.num_processes):
        # p = mp.Process(target = train, args = (agent, shared_ac, args, device, idx, counter, lock))
        p = mp.Process(target = train, args = (agent, shared_ac, args, device, idx, counter, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()




