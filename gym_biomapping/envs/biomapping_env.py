import gym
from gym import error, spaces, utils
from gym.utils import seeding

import matplotlib.pyplot as plt
import numpy as np

from gym_biomapping.envs.env_sim import EnvSim
from gym_biomapping.envs.auv_sim import AUVsim

from pathlib import Path

DATA_DIR = Path(__file__).parent.joinpath('data')


class BioMapping(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dt=60, pos0=None, data_file='bio2d_v2_samples_TrF_2018.04.27.nc',
                 static=False, n_auvs=1):
        super(BioMapping, self).__init__()
        assert (dt > 0)
        self.dt = dt
        self.t = 0.0
        self.env_sim = EnvSim(data_file=data_file, exact_indexed=False, static=static, n_auvs=n_auvs)
        if pos0 is None:
            pos0 = self.env_sim.pos_space.sample()

        self.auv_sim = AUVsim(pos0, speed=1.0, dt=dt)
        self.action_space = self.env_sim.pos_space
        self.observation_space = spaces.Dict({"pos": self.env_sim.pos_space, "env": self.env_sim.env_space})
        self.fig, self.ax1 = plt.subplots()
        self.reset()

    def step(self, action):
        self.t += self.dt
        pos = self.auv_sim.step(action)
        env_state, measurement = self.env_sim.step(pos, self.t)
        state = {"pos": pos, "env": env_state}
        reward = np.linalg.norm(measurement, axis=0)
        done = False
        info = {}
        return state, reward, done, info

    def reset(self):
        self.t = 0.0
        return {"pos": self.auv_sim.reset(), "env": self.env_sim.reset()}

    def render(self, mode='human'):
        self.ax1.clear()
        self.ax1.set_title("Simulation duration: " + str(self.t) + "s")
        self.env_sim.render(self.ax1)
        self.auv_sim.render(self.ax1)
        self.fig.canvas.draw()

    def close(self):
        return
