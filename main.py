import gym
import torch
import torch.nn as nn
import torch.distributions as dist
from collections import deque

import random
import numpy as np
torch.manual_seed(0)
np.random.seed(0)
agent = nn.Sequential(
    nn.Linear(4, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 2),
    nn.LogSoftmax(),
)

critic = nn.Sequential(
    nn.Linear(4, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid(),
)
optim = torch.optim.Adam(list(agent.parameters()) + list(critic.parameters()), lr=1e-3)
agent.cuda()
critic.cuda()
env = gym.make("CartPole-v1")
EPISODES = 10000
DISCOUNT_RATE = 0.97
scores = deque(maxlen=100)
for i in range(EPISODES):
    episode = []

    # Go through an episode...
    observation = env.reset()
    observation = torch.Tensor(observation).cuda()
    for j in range(1000):
        #env.render()

        log_probs = agent.forward(observation)
        distribution = dist.Categorical(probs=torch.exp(log_probs))
        action = int(distribution.sample())  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        observation = torch.Tensor(observation).cuda()
        state_dict = {
            'obs': observation,
            'reward': reward,
            'action': action,
            'action_log_prob': log_probs[action]
        }
        episode.append(state_dict)
        if done:
            break
    scores.append(j)
    if np.mean(scores) >= 195.0 and i >= 100:
        print("Solved in {} episodes.".format(i - 100))
        break
    # Discount rewards...
    running_reward = 0
    for state_dict in episode[::-1]:
        state_dict['reward'] += DISCOUNT_RATE * running_reward
        running_reward = state_dict['reward']

    # Train...
    optim.zero_grad()
    action_log_probs = torch.stack([x['action_log_prob'] for x in episode]).cuda()
    returns = torch.Tensor([x['reward'] for x in episode]).cuda()
    observations = torch.stack([x['obs'] for x in episode]).cuda()

    values = 0#critic.forward(observations)

    returns -= returns.mean()
    returns /= returns.std()



    loss = (- (returns - values).detach() * action_log_probs).sum() #+ torch.pow((returns - values), 2).mean()
    loss.backward()
    optim.step()
    print("Episode {}, Avg score of last hundred eps {}".format(i, np.mean(scores)))


observation = env.reset()
for j in range(1000):
    env.render()
    observation = torch.Tensor(observation).cuda()
    log_probs = agent.forward(observation)
    #distribution = dist.Categorical(probs=torch.exp(log_probs))
    #action = int(distribution.sample())  # your agent here (this takes random actions)
    action = int(torch.argmax(log_probs))
    observation, reward, done, info = env.step(action)
    state_dict = {
        'obs': observation,
        'reward': reward,
        'action': action,
        'action_log_prob': log_probs[action]
    }
    episode.append(state_dict)
    if done:
        break
env.close()
