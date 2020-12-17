import gym
from gym import error, spaces, utils
from gym.utils import seeding

import matplotlib.pyplot as plt
import numpy as np

from gym_biomapping.envs.env_sim import EnvSim
from gym_biomapping.envs.auv_sim import AUVsim, AtariAUVSim

from pathlib import Path

DATA_DIR = Path(__file__).parent.joinpath('data')

def _construct_output(env, pos, output_format):
    if output_format == 'dict':
        state = {"pos": pos, "env": env}
    elif output_format == '3D-matrix':
        img_pos = np.zeros(env.shape)
        img_pos[pos[0, 0], pos[0, 1]] = 1
        state = np.stack([env, img_pos], axis=2)
    else:
        raise ValueError("Unsupported output format {!r]".format(output_format))
    return state

class AtariBioMapping(gym.Env):
    
    
    def __init__(self, pos0=None, data_file='bio2d_v2_samples_TrF_2018.04.27.nc', static=False, output='dict', n_auvs=1):
        super(AtariBioMapping, self).__init__()
        assert n_auvs == 1, "Currently not implemented support for multiple AUVs"
        self.t = 0.0
        self.env_sim = EnvSim(data_file=data_file, exact_indexed=True, static=static, n_auvs=n_auvs)
        if pos0 is None:
            pos0 = np.array(self.env_sim.pos_space.sample(), dtype=np.int).reshape((1, 3))
        self.STATE_SPACE_SHAPE = (50,50,2)
        self.auv_sim = AtariAUVSim(pos0, speed=1.0, square_size=50, sample_time=120, n_auvs=n_auvs)
        self.action_space = self.auv_sim.action_space
        self.observation_space = spaces.Dict({"pos": self.env_sim.pos_space, "env": self.env_sim.env_space})
        self.fig, self.ax1 = plt.subplots()
        self.output = output
        self.MAX_STEPS = 50 # Steps to take before the episode is done: Should reflect length of a mission
        self.steps_taken = 0
        self.reset()

    def step(self, action):
        pos, dt = self.auv_sim.step(action)
        self.t += dt[0]  # Currently not support for multiple AUVs in atari mode as,
        # with the current implementation, it would require
        # keeping track of one timeline for each AUV, making plotting difficult and
        # having to interpolate env data in time for each auv.
        env_state, measurement = self.env_sim.step(pos, self.t)
        state = _construct_output(env_state, pos, self.output)
        # Check if agent is out-of-bounds
        if any(pos[0][0:1] > (50, 50)) or any(pos[0][0:1] < (0, 0)):
            print("AUV is out of bounds")
            reward = -1000
            done = True
            info = {}
            return state, reward, done, info
        reward = np.linalg.norm(measurement, axis=0) if action == [0] else 0
        if self.steps_taken >= self.MAX_STEPS:
            print("Episode reached max number of steps")
            done = True
        else:
            done = False
        info = {}
        self.steps_taken += 1
        return state, reward, done, info

    def reset(self):
        self.t = 0.0
        self.steps_taken = 0
        return _construct_output(self.env_sim.reset(), self.auv_sim.reset(), self.output)

    def render(self, mode='human'):
        self.ax1.clear()
        self.ax1.set_title("Simulation duration: " + str(self.t) + "s")
        self.env_sim.render(self.ax1)
        self.auv_sim.render(self.ax1)
        self.fig.canvas.draw()

    def close(self):
        return


class BioMapping(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dt=60, pos0=None, data_file='bio2d_v2_samples_TrF_2018.04.27.nc',
                 static=False, output='dict', n_auvs=1):
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
        self.output = output
        self.reset()

    def step(self, action):
        self.steps_taken +=1
        self.t += self.dt
        pos = self.auv_sim.step(action)
        env_state, measurement = self.env_sim.step(pos, self.t)
        state = _construct_output(env_state, pos, self.output)
        reward = np.linalg.norm(measurement, axis=0)
        if self.steps_taken >= stelf.MAX_STEPS:
            print("Episode reached max number of steps")
            done = True
        else:
            print("On step number :", self.steps_taken)
            done = False
        info = {}
        return state, reward, done, info

    def reset(self):
        self.t = 0.0
        return _construct_output(self.env_sim.reset(), self.auv_sim.reset(), self.output)

    def render(self, mode='human'):
        self.ax1.clear()
        self.ax1.set_title("Simulation duration: " + str(self.t) + "s")
        self.env_sim.render(self.ax1)
        self.auv_sim.render(self.ax1)
        self.fig.canvas.draw()

    def close(self):
        return
