import numpy as np
import matplotlib.pyplot as plt
import cmocean

"""

class SimpleAtariAgent:

    def __init__(self, render=True, input='dict'):
        if render:
            self.fig, self.ax = plt.subplots()
        self.g_pos = np.zeros(3)
        self.input = input

    def deliberate(self, obs):
        if self.input == 'dict':
            pos = obs['pos'][0]
            env = obs["env"]
        if self.input == 'flatten':
            pos = obs[-3:]
            env = obs[:-3].reshape(50, 50)
        if self.input == '3D-matrix':
            pos = np.zeros(3)
            pos[:2] = np.array(np.nonzero(obs[:, :, 1])).flatten()
            env = obs[:, :, 0]
        ind = np.unravel_index(np.argmax(env), env.shape)
        g_pos = np.array([ind[0], ind[1], 0])
        action = [0, 0, 0]
        for i in range(2):
            if pos[i] < g_pos[i]:
                action[i] = 1
            if pos[i] > g_pos[i]:
                action[i] = -1

        actions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0],
                  [-1, 0, 0], [-1, -1, 0], [0, -1, 0], [1, -1, 0]]
        index = actions.index(action)
        self.g_pos = g_pos
        return [index]

    def render(self, obs, next_pos):
        self.ax.clear()
        self.ax.set_title('Observation')
        if self.input == 'dict':
            self.ax.pcolormesh(obs["env"].T,
                               cmap=cmocean.cm.deep,
                               shading='auto')
            self.ax.plot(self.g_pos[0], self.g_pos[1], 'go')
        if self.input == '3D-matrix':
            self.ax.pcolormesh(obs[:, :, 1].T)
            self.ax.plot(self.g_pos[0], self.g_pos[1], 'go')
        self.fig.canvas.draw()

class SimpleAgent:

    def __init__(self, xc, yc, render=True, n_auvs=1):
        self.totalscore = 0
        self.xc = xc  # Mapping from index to x coordinate
        self.yc = yc  # Mapping from index to y coordinate
        self.n_auvs = n_auvs
        if render:
            self.fig, self.ax = plt.subplots()

    def deliberate(self, obs):
        ind = np.unravel_index(np.argmax(obs["env"]), obs["env"].shape)
        best_pos = np.array([self.xc[ind[0]], self.yc[ind[1]], 0])
        next_pos = np.repeat(best_pos[np.newaxis, :], self.n_auvs, axis=0)
        return next_pos

    def render(self, obs, next_pos):
        self.ax.clear()
        self.ax.set_title('Observation')
        self.ax.pcolormesh(self.xc, self.yc, obs["env"].T,
                           cmap=cmocean.cm.deep,
                           shading='auto')
        self.ax.plot(next_pos[:, 0], next_pos[:, 1], 'go')
        self.fig.canvas.draw()
"""