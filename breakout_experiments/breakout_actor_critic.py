import gym
import torch
import torch.nn as nn
import torchvision
import episode_runner
import actor_critic_agent


class BreakoutA2CAgent(actor_critic_agent.A2CAgent, nn.Module):

    def __init__(self):
        super(actor_critic_agent.A2CAgent, self).__init__()
        super(nn.Module, self).__init__()

        self.conv1 = nn.Conv2d(2, 16, 9, 4)
        self.conv2 = nn.Conv2d(16, 32, 9, 2)
        self.fc1 = nn.Linear(1152, 256)
        self.actor = nn.Linear(256, 6)
        self.value = nn.Linear(256, 1)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        self.optim = torch.optim.Adam(self.parameters())
        self.activation = nn.ReLU()

        env = gym.make("PongNoFrameskip-v4")
        self.episode_runner = episode_runner.EpisodeRunner(env, self, self.to_tensor)

        self.resize = torchvision.transforms.Resize((84, 84))
        self.grayscale = torchvision.transforms.Grayscale()

    def forward(self, x):
        x = x.permute(2, 0, 1)

        x = self.grayscale(x)
        x = self.resize(x)
        x /= 255.0
        temp = x.clone()
        if self.last_input is not None:
            x = torch.cat([x, self.last_input], dim=0)
        else:
            x = torch.cat([x, x], dim=0)
        self.last_input = temp

        x = torch.unsqueeze(x, 0)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)

        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = self.activation(x)
        log_probs = self.actor(x)
        log_probs = nn.LogSoftmax()(log_probs)
        value = self.value(x)
        return log_probs, value

    def reset(self):
        self.last_input = None


if __name__ == '__main__':
    agent = BreakoutA2CAgent()
    agent.run(100000, discount_rate=0.99)
    agent.episode_runner.run_episode(render=True, greedy=True)
