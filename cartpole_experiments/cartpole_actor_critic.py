import gym
import torch
import torch.nn as nn

import episode_runner
import actor_critic_agent


class CartpoleA2CAgent(actor_critic_agent.A2CAgent, nn.Module):

    def __init__(self):
        super(actor_critic_agent.A2CAgent, self).__init__()
        super(nn.Module, self).__init__()

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.actor = nn.Linear(64, 2)
        self.value = nn.Linear(64, 1)
        self.activation = nn.ReLU()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)

        env = gym.make("CartPole-v1")
        self.episode_runner = episode_runner.EpisodeRunner(env, self, self.to_tensor)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        log_probs = self.actor(x)
        log_probs = nn.LogSoftmax()(log_probs)
        value = self.value(x)
        value = nn.Sigmoid()(value)
        return log_probs, value


if __name__ == '__main__':
    agent = CartpoleA2CAgent()
    agent.run(1000, discount_rate=0.99)
    agent.episode_runner.run_episode(render=True, greedy=True)
