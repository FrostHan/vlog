import numpy as np

from minatar import Environment, GUI
import gym
from gym.spaces import Discrete, Box


class MinAtarEnv(gym.Env):
    def __init__(self, env_id='breakout', sticky_action_prob=0.1, difficulty_ramping=True, random_seed=None,
                 broken_pixel_mask=None, broken_pixels_ratio=0):

        self.sticky_action_prob = sticky_action_prob
        self.difficulty_ramping = difficulty_ramping

        self.env_id = env_id
        self.game = Environment(env_id, sticky_action_prob=sticky_action_prob,
                                difficulty_ramping=difficulty_ramping, random_seed=random_seed)

        self.observation_space = Box(0, 1, shape=[self.game.n_channels, 10, 10])
        self.oracle_observation_space = self.observation_space
        self.action_space = Discrete(self.game.num_actions())

        if broken_pixel_mask is None:
            self.broken_pixels_ratio = broken_pixels_ratio
            self.broken_pixel_mask = np.random.choice(2, size=[1, 10, 10],
                                                      p=[1 - self.broken_pixels_ratio, self.broken_pixels_ratio])
            self.random_broken_pixels = True
        else:
            self.broken_pixel_mask = broken_pixel_mask
            self.broken_pixels_ratio = np.sum(broken_pixel_mask) / 100
            self.random_broken_pixels = False

    def seed(self, seeding=None):
        self.game = Environment(self.env_id, sticky_action_prob=self.sticky_action_prob,
                                difficulty_ramping=self.difficulty_ramping, random_seed=seeding)

    def reset(self):
        self.game.reset()
        obs = self.game.state().astype(np.float32).swapaxes(0, 2).swapaxes(2, 1)
        if self.broken_pixels_ratio:
            if self.random_broken_pixels:
                broken_pixel_mask = np.random.choice(2, size=[1, 10, 10],
                                                     p=[1 - self.broken_pixels_ratio, self.broken_pixels_ratio])
            else:
                broken_pixel_mask = self.broken_pixel_mask
            obs = obs - (obs * broken_pixel_mask) + 2 * broken_pixel_mask  # broken pixels obs = 2, normal obs = 0 or 1

        return obs

    def get_oracle_obs(self):
        return self.game.state().astype(np.float32).swapaxes(0, 2).swapaxes(2, 1)

    def get_full_obs(self):
        return np.concatenate([self.get_obs(), self.get_oracle_obs()], axis=0)

    def get_obs(self):
        obs = self.game.state().astype(np.float32).swapaxes(0, 2).swapaxes(2, 1)
        if self.broken_pixels_ratio:
            if self.random_broken_pixels:
                broken_pixel_mask = np.random.choice(2, size=[1, 10, 10],
                                                     p=[1 - self.broken_pixels_ratio, self.broken_pixels_ratio])
            else:
                broken_pixel_mask = self.broken_pixel_mask
            obs = obs - (obs * broken_pixel_mask) + 2 * broken_pixel_mask  # broken pixels obs = 2, normal obs = 0 or 1
        return obs

    def step(self, action):
        reward, done = self.game.act(action)
        obs = self.get_obs()
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
