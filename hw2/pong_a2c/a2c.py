import torch
import torch.nn as nn
import torch.optim as optim


class A2C():
    def __init__(self, model, args):
        self.model = model
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:] # squeeze
        num_steps, action_shape = rollouts.actions.size()
        values, action_log_probs, dist_entropy = self.model.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, 1)
        action_log_probs = action_log_probs.view(num_steps, 1)

        advantages = rollouts.qval[:-1] - values
        value_loss = advantages.pow(2).mean()

        policy_loss = -(advantages.detach() * action_log_probs).mean()

        
        total_loss = value_loss*self.value_loss_coef + policy_loss - dist_entropy*self.entropy_coef
        
        return total_loss
