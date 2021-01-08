import torch
import numpy as np

class Agent:

    def get_action(self, observation, greedy):
        pass

    def reset(self):
        pass

    def discount_rewards(self, episode, discount_rate=0.99):
        running_return = 0
        for state_dict in episode[::-1]:
            state_dict['reward'] += discount_rate * running_return
            running_return = state_dict['reward']
        return episode

    @staticmethod
    def to_tensor(obs):
        return torch.Tensor(obs)

    @staticmethod
    def sum_rewards(episode):
        return sum([state['reward'] for state in episode])
