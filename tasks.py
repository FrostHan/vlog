import gym
from gym import spaces, logger
import numpy as np
from gym.utils import seeding
from copy import deepcopy
import warnings
import os
import gym_maze


class SimpleMaze(gym.Env):
    def __init__(self, use_vision=False, noisy=True):
        self.env = gym.make("maze-sample-10x10-v0")
        self.action_space = self.env.action_space
        self.use_vision = use_vision
        self.noisy = noisy

        if not self.use_vision:
            self.observation_space = self.env.observation_space
            self.oracle_observation_space = gym.spaces.Box(0, 10, [3])
        else:
            self.observation_space = gym.spaces.Box(0, 1, [3, 10, 10])
            self.oracle_observation_space = gym.spaces.Box(0, 1, [3, 10, 10])

        self.dis2goal = np.array([[62, 59, 58, 57, 56, 55, 54, 15, 14, 13],
                                  [61, 60, 59, 40, 41, 42, 53, 10, 11, 12],
                                  [62, 61, 60, 39, 40, 43, 52,  9,  8,  7],
                                  [63, 62, 31, 38, 41, 42, 51, 50, 49,  6],
                                  [28, 29, 30, 37, 16, 43, 44, 45, 48,  5],
                                  [27, 30, 35, 36, 15, 14, 13, 46, 47,  4],
                                  [26, 31, 34, 15, 14, 13, 12, 11, 10,  3],
                                  [25, 32, 33, 16, 15,  8,  7,  8,  9,  2],
                                  [24, 23, 20, 19, 16,  7,  6,  3,  2,  1],
                                  [23, 22, 21, 18, 17,  6,  5,  4,  1,  0]], dtype=np.float32)

        self.dis2goal = self.dis2goal / np.max(self.dis2goal)
        self.max_steps = 5000

    def reset(self):
        self.env.reset()
        return self.get_obs()

    def step(self, a):
        s, r, d, _ = self.env.step(a)
        info = np.zeros([3], dtype=np.float32)
        info[0] = self.env.maze_view.robot[0]
        info[1] = self.env.maze_view.robot[1]
        info[2] = self.dis2goal[int(info[0]), int(info[1])]
        return self.get_obs(), r, d, info

    def render(self):
        return self.env.render().astype(np.float32) / 255.0

    def close(self):
        self.env.close()

    def get_local_vision(self, x, y):
        img = self.render()
        padded = np.zeros([250 + 640, 250 + 640, 3], dtype=np.float32)
        padded[125: -125, 125: -125, :] = img

        img = padded[y * 64 + 32: y * 64 + 32 + 250, x * 64 + 32: x * 64 + 32 + 250, :]

        img = img.reshape([10, 25, 250, 3])
        img = img.mean(axis=1)

        img = img.reshape([10, 10, 25, 3])
        img = img.mean(axis=2)

        return img.swapaxes(0, 2)

    @ staticmethod
    def avg_pooling(img):
        img = img.reshape([10, 64, 640, 3])
        img = img.mean(axis=1)

        img = img.reshape([10, 10, 64, 3])
        img = img.mean(axis=2)
        return img.swapaxes(0, 2)

    def get_obs(self):
        if not self.use_vision:
            if self.noisy:
                return self.env.maze_view.robot + np.random.uniform(-0.5, 0.5, [2])
            else:
                return self.env.maze_view.robot
        else:
            xy = self.env.maze_view.robot
            return self.get_local_vision(xy[0], xy[1])

    def get_oracle_obs(self):
        if not self.use_vision:
            state = self.env.maze_view.robot
            ind1 = int(state[0])
            ind2 = int(state[1])
            return np.concatenate([state, np.array([self.dis2goal[ind1][ind2]])])
        else:
            return self.avg_pooling(self.render())

