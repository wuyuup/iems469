import torch
import random
from torch.autograd import Variable

class ReplayMemory(object):
    """ Facilitates memory replay. """
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.memory = []
        self.idx = 0
        self.device = device

    def push(self, m):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.idx] = m
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, bsz):
        batch = random.sample(self.memory, bsz)
        return map(lambda x: Variable(torch.cat(x, 0).to(self.device)), zip(*batch))
    def __len__(self):
        return len(self.memory)
