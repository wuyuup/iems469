import torch

class RolloutBuffer(object):
    def __init__(self, num_steps, obs_shape, action_space, device):
        self.obs = torch.zeros(num_steps + 1, *obs_shape).to(device)
        self.rewards = torch.zeros(num_steps, 1).to(device)
        self.value_preds = torch.zeros(num_steps + 1, 1).to(device)
        self.qval = torch.zeros(num_steps + 1, 1).to(device)
        self.action_log_probs = torch.zeros(num_steps, 1).to(device)
        self.actions = torch.zeros(num_steps, 1).to(device)
        self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, 1).to(device)
        self.num_steps = num_steps
        self.step = 0


    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1] = obs
        self.actions[self.step] = actions
        self.action_log_probs[self.step] = action_log_probs
        self.value_preds[self.step] = value_preds
        self.rewards[self.step] = torch.Tensor([rewards])
        self.masks[self.step + 1] = masks
        self.step = (self.step + 1) % self.num_steps

    def recover_state(self):
        self.obs[0] = self.obs[-1]
        self.masks[0] = self.masks[-1]

    def compute_qval(self,next_value,args):
        self.qval[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.qval[step] = self.qval[step + 1] * args.gamma * self.masks[step + 1] + self.rewards[step]
