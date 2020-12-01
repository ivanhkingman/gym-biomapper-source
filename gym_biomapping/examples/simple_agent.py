import numpy as np
import matplotlib.pyplot as plt
import cmocean


class SimpleAgent:

    def __init__(self, xc, yc, render=True):
        self.totalscore = 0
        self.xc = xc  # Mapping from index to x coordinate
        self.yc = yc  # Mapping from index to y coordinate
        if render:
            self.fig, self.ax = plt.subplots()

    def deliberate(self, obs):
        ind = np.unravel_index(np.argmax(obs["env"]), obs["env"].shape)
        next_pos = np.array([[self.xc[ind[0]]], self.yc[[ind[1]]], [0]])
        return next_pos

    def render(self, obs, next_pos):
        self.ax.clear()
        self.ax.set_title('Observation')
        self.ax.pcolormesh(self.xc, self.yc, obs["env"].T,
                           cmap=cmocean.cm.deep,
                           shading='auto')
        self.ax.plot(next_pos[0], next_pos[1], 'go')
        self.fig.canvas.draw()
