import torch
import torch.distributions as dist
import numpy as np
from collections import deque
import agent


class A2CAgent(agent.Agent):

    def __init__(self):
        self.device = None
        self.optim = None
        self.episode_runner = None

    def forward(self):
        raise NotImplementedError("Implement a function that returns the log "
                                  "probs of the policy and the value of the agent")

    def get_action(self, observation, greedy=False):
        observation = observation.to(self.device)
        log_probs, value = self.forward(observation)
        log_probs = log_probs.view(-1)
        if greedy:
            action = torch.argmax(log_probs).detach().cpu()
        else:
            cat_dist = dist.Categorical(probs=torch.exp(log_probs))
            action = cat_dist.sample()
        action_info = {
            'value': value,
            'log_probs': log_probs,
            'action_log_prob': log_probs[action]
        }
        return int(action), action_info

    def train(self, episode, discount_rate=0.99):
        self.discount_rewards(episode, discount_rate)

        self.optim.zero_grad()
        rewards = torch.Tensor([state['reward'] for state in episode]).to(self.device)
        action_infos = [state['action_info'] for state in episode]
        values = torch.stack([action_info['value'] for action_info in action_infos]).to(self.device)
        action_log_probs = torch.stack([action_info['action_log_prob'] for action_info in action_infos]).to(self.device)
        log_probs = torch.stack([action_info['log_probs'] for action_info in action_infos]).to(self.device)
        rewards -= rewards.mean()
        rewards /= rewards.std() + 1e-6
        adv = rewards - values.detach()

        policy_loss = (-adv * action_log_probs).mean()

        value_loss = torch.nn.SmoothL1Loss()(values.view(-1), rewards)
        entropy_loss = (log_probs * torch.exp(log_probs)).mean()
        loss = policy_loss + value_loss + 1e-5 * entropy_loss
        loss.backward()
        self.optim.step()

        return policy_loss, value_loss

    def run(self, num_episodes, avg_threshold=195.0, discount_rate=0.99):
        scores = deque(maxlen=100)

        for episode_idx in range(1, num_episodes + 1):
            episode = self.episode_runner.run_episode(render=True, greedy=False)
            score = self.sum_rewards(episode)
            scores.append(score)
            p, v = self.train(episode, discount_rate)

            avg_score = np.mean(scores)

            print("At episode: {}, avg of last 100 episodes: {}".format(episode_idx, avg_score))
            #print("Policy Loss: {}, Value Loss: {}".format(p, v))
            if avg_score >= avg_threshold and episode_idx > 100:
                print("Solved in {} episodes".format(episode_idx))
                break
