import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, obs_shape, hidden_size=512):
        super(Policy, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1), nn.ReLU())

        self.linear = nn.Sequential(nn.Linear(32 * 6 * 6, hidden_size), nn.ReLU())
        self.actor_linear = nn.Linear(hidden_size, 2) # output a 
        self.critic_linear = nn.Linear(hidden_size, 1) # output val
        self.train() # set to train mode

    def val_and_feature(self, inputs):
        x = self.feature(inputs)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return self.critic_linear(x), self.actor_linear(x)

    def forward(self, inputs):
        value, logits = self.val_and_feature(inputs)
        action_category = Categorical(logits=logits)
        action = action_category.sample().unsqueeze(-1) # left and right
        action_log_probs = action_category.log_prob(action.squeeze(-1)).view(action.size(0), -1).sum(-1).unsqueeze(-1)
        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.val_and_feature(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, logits = self.val_and_feature(inputs)
        action_category = Categorical(logits=logits)
        action_log_probs = action_category.log_prob(action.squeeze(-1)).view(action.size(0), -1).sum(-1).unsqueeze(-1)
        dist_entropy = action_category.entropy().mean()
        return value, action_log_probs, dist_entropy
