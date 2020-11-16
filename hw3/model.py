import torch.nn.functional as F
from torch import nn
import torch

class ValueNetwork(nn.Module):
    def __init__(self, action_dim, env, hidden_layers=512):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        if env.unwrapped.spec.id == 'Breakout-v0':
            self.fc1 = nn.Linear(6 * 6 * 64, hidden_layers)
        elif env.unwrapped.spec.id == 'MsPacman-v0':
            self.fc1 = nn.Linear(6 * 7 * 64, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, action_dim)


    def forward(self, x):
        # print(x, x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)
