import numpy as np
import matplotlib.pyplot as plt
import cmocean

class SimpleAtariAgent:

    def __init__(self, render=True):
        if render:
            self.fig, self.ax = plt.subplots()
        self.g_pos = np.zeros(3)

    def deliberate(self, obs):
        ind = np.unravel_index(np.argmax(obs["env"]), obs["env"].shape)
        g_pos = np.array([ind[0], ind[1], 0])
        pos = obs['pos'][0]
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
        self.ax.pcolormesh(obs["env"].T,
                           cmap=cmocean.cm.deep,
                           shading='auto')
        self.ax.plot(obs['pos'][:, 0], obs['pos'][:, 1], 'go')
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
