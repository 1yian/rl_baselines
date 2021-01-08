import gym
import torch
import torch.nn as nn
import torchvision
import episode_runner
import actor_critic_agent


class PongA2CAgent(actor_critic_agent.A2CAgent, nn.Module):

    def __init__(self):
        super(actor_critic_agent.A2CAgent, self).__init__()
        super(nn.Module, self).__init__()

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.actor = nn.Linear(64, 4)
        self.value = nn.Linear(64, 1)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        self.optim = torch.optim.Adam(self.parameters())
        self.activation = nn.ReLU()

        env = gym.make("Breakout-ram-v4")
        print(env.action_space)
        self.episode_runner = episode_runner.EpisodeRunner(env, self, self.to_tensor)

        self.resize = torchvision.transforms.Resize((84, 84))
        self.grayscale = torchvision.transforms.Grayscale()

    def forward(self, x):
        x /= 255.0
        print(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        log_probs = self.actor(x)
        log_probs = nn.LogSoftmax()(log_probs)
        value = self.value(x)
        print(value)
        return log_probs, value

    def reset(self):
        self.last_input = None


if __name__ == '__main__':
    agent = PongA2CAgent()
    agent.run(100000, discount_rate=0.99)
    agent.episode_runner.run_episode(render=True, greedy=True)
