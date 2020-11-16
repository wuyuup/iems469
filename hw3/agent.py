from itertools import count
from torch import optim
import torch
from torch import nn
import numpy as np
import time
from itertools import count
from torch.autograd import Variable
import logging
import os

from utils import preprocess_state
from model import ValueNetwork
from buffer import ReplayMemory


class DQNAgent(object):
    def __init__(self, env, args, work_dir):
        self.env = env
        self.args = args
        self.work_dir = work_dir

        self.n_action = self.env.action_space.n
        self.arr_actions = np.arange(self.n_action)
        self.memory = ReplayMemory(self.args.buffer_size, self.args.device)
        self.qNetwork = ValueNetwork(self.n_action, self.env).to(self.args.device)
        self.targetNetwork = ValueNetwork(self.n_action, self.env).to(self.args.device)
        self.qNetwork.train()
        self.targetNetwork.eval()
        self.optimizer = optim.RMSprop(self.qNetwork.parameters(),lr=0.00025, eps=0.001, alpha=0.95)
        self.crit = nn.MSELoss()
        self.eps = max(self.args.eps, self.args.eps_min)
        self.eps_delta = (self.eps - self.args.eps_min) / self.args.exploration_decay_speed

    def reset(self):
        return torch.cat([preprocess_state(self.env.reset(), self.env)] * 4, 1)

    def select_action(self, state):
        action_prob = np.zeros(self.n_action, np.float32)
        action_prob.fill(self.eps / self.n_action)
        max_q, max_q_index = self.qNetwork(Variable(state.to(self.args.device))).data.cpu().max(1)
        action_prob[max_q_index[0]] += 1 - self.eps
        action = np.random.choice(self.arr_actions, p = action_prob)
        next_state, reward, done, _ = self.env.step(action)
        next_state = torch.cat([state.narrow(1, 1, 3), preprocess_state(next_state, self.env)], 1)
        self.memory.push((state, torch.LongTensor([int(action)]),
            torch.Tensor([reward]), next_state, torch.Tensor([done])))
        return next_state, reward, done, max_q[0]



    def run(self):
        state = self.reset()
        # init buffer
        for _ in range(self.args.buffer_init_size):
            next_state, _, done, _ = self.select_action(state)
            state = self.reset() if done else next_state
    
        total_frame = 0
        reward_list = np.zeros(self.args.log_size, np.float32)
        qval_list = np.zeros(self.args.log_size, np.float32)
        
        start_time = time.time()
        
        for epi in count():
            reward_list[epi % self.args.log_size] = 0
            qval_list[epi % self.args.log_size] = -1e9
            state = self.reset()
            done = False
            ep_len = 0
    
            if epi % self.args.save_freq == 0:
                model_file = os.path.join(self.work_dir, 'model.th')
                with open(model_file, 'wb') as f:
                    torch.save(self.qNetwork, f)
    
            while not done:
                if total_frame % self.args.sync_period == 0:
                    self.targetNetwork.load_state_dict(self.qNetwork.state_dict())
    
                self.eps = max(self.args.eps_min, self.eps - self.eps_delta)
                next_state, reward, done, qval = self.select_action(state)
                reward_list[epi % self.args.log_size] += reward
                qval_list[epi % self.args.log_size] = max(qval_list[epi % self.args.log_size], qval)
                state = next_state
    
                total_frame += 1
                ep_len += 1

                if ep_len % self.args.learn_freq == 0:
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.memory.sample(self.args.batch_size)
                    batch_q = self.qNetwork(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
                    batch_next_q = self.targetNetwork(batch_next_state).detach().max(1)[0] * self.args.gamma * (1 - batch_done)
                    loss = self.crit(batch_q, batch_reward + batch_next_q)
                    self.optimizer.zero_grad()
    
                    loss.backward()
                    self.optimizer.step()
    
            output_str = 'episode %d frame %d time %.2fs cur_rew %.3f mean_rew %.3f cur_maxq %.3f mean_maxq %.3f' % (
                epi, total_frame, time.time() - start_time,
                reward_list[epi % self.args.log_size], np.mean(reward_list),
                qval_list[epi % self.args.log_size], np.mean(qval_list))
            print(output_str)
            logging.info(output_str)
    
    