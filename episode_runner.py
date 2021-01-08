import gym
import agent


class EpisodeRunner:

    def __init__(self, env: gym.Env, agent: agent.Agent, preprocess_state=None):
        self.preprocess = preprocess_state
        self.env = env
        self.agent = agent

    def run_episode(self, render=False, max_steps=100_000, greedy=False) -> list:
        episode = []
        observation = self.env.reset()
        self.agent.reset()
        if self.preprocess:
            observation = self.preprocess(observation)

        for _ in range(max_steps):
            if render:
                self.env.render()

            action, action_info = self.agent.get_action(observation, greedy)
            observation, reward, done, info = self.env.step(action)
            observation = self.preprocess(observation)
            state_dict = {
                'obs': observation,
                'reward': reward,
                'done': done,
                'action_info': action_info,
            }

            episode.append(state_dict)

            if done:
                break

        return episode


