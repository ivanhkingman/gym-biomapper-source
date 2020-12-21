import pyproj
import numpy as np
from gym import spaces, error


class AtariAUVSim:
    def __init__(self, pos0=None, space=None, speed=1.0, square_size=50, sample_time=120, n_auvs=1):
        assert speed > 0
        assert sample_time > 0
        assert square_size > 0
        self.pos0 = pos0

        self.state_space = space
        self.action_space = spaces.MultiDiscrete(np.ones(n_auvs)*9)
        self.action_realization = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (-1, 1, 0),
                                            (-1, 0, 0), (-1, -1, 0), (0, -1, 0), (1, -1, 0)], dtype=np.int)
        self.action_duration = np.array([np.linalg.norm(x) * square_size / speed for x in self.action_realization])
        self.action_duration[0] = sample_time

    def step(self, action):
        """
        Do one simulation step
        :param action a positive int in the range [0, 9], with the following meaning;
        0:sample, 1:north, 2:nort-east, 3:east, 4:south-east, 5:south,
        6:south-west, 7:west, 8:north-west.
        :return pos the position of the AUVs after executing the action.
        :return time the time [s] the action took to execute.
        """
        if not self.action_space.contains(action):
            raise error.InvalidAction()
        self.pos += self.action_realization[action]
        time = self.action_duration[action]
        return self.pos, time

    def reset(self):
        if self.pos0 is None:
            self.pos = np.array(self.state_space.sample(), dtype=np.int).reshape((1, 3))
        else:
            self.pos = self.pos0
        return self.pos

    def render(self, ax):
        ax.plot(self.pos[:, 0], self.pos[:, 1], 'ro')


class AUVsim:

    # metadata = {
    #    'render.modes': ['human', 'rgb_array'],
    #    'plot.frames_per_second': 50
    # }

    def __init__(self, pos0=np.zeros((1, 3)), speed=1.0, dt=1.0, xy=None, lonlat=None):
        if not isinstance(pos0, np.ndarray):
            pos0 = np.array(pos0)
        assert pos0.shape[1] == 3, "Position must have n_auvs x 3 shape"
        n_auvs = pos0.shape[0]
        if np.size(speed) != 1:
            assert np.size(speed) == n_auvs, "The AUVs must have the same speed or one each"
            assert all(speed > 0), "All auv speeds must be positive"
        else:
            assert speed > 0, "Speed must be positive"
        assert dt > 0
        self.pos0 = pos0
        self.pos = pos0
        self.distance = speed * dt  # m/s
        self.xy = xy
        self.lonlat = lonlat

    def step(self, action):
        """
        Do one simulation step
        :param action  is Nx3 matrix, where N is the number of AUVs
        action[5][0] returns x value of the goal position for auv number 6.
        :return state, current position of the AUVs, same structure as action.
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        assert action.shape == self.pos.shape
        # goal = np.zeros((3, self.nauvs))
        # goal[0], goal[1] = self.lonlat2xy(action[0], action[1])

        dv = action - self.pos
        dist = np.linalg.norm(dv, axis=1, keepdims=True)

        at_goal = (dist < self.distance).flatten()
        self.pos[at_goal] = action[at_goal]
        self.pos[~at_goal] += (dv[~at_goal] / dist[~at_goal]) * self.distance

        # self.heading = np.arctan2(dv[1], dv[0])

        # lon, lat = self.xy2lonlat(self.pos[0], self.pos[1])
        # return np.array([lon, lat, self.pos[2]])
        return self.pos

    def reset(self):
        self.pos = self.pos0
        return self.pos

    def render(self, ax):
        ax.plot(self.pos[:, 0], self.pos[:, 1], 'ro')

    def xy2lonlat(self, x, y):
        # Convert from x-y coordinates to lon-lat
        # (this function works on both scalars and arrays)
        if self.xy and self.lonlat:
            return pyproj.transform(self.xy, self.lonlat, x, y)
        else:
            return x, y

    def lonlat2xy(self, lon, lat):
        # Convert from lon-lat to x-y coordinates
        # (this function works on both scalars and arrays)
        if self.xy and self.lonlat:
            return pyproj.transform(self.lonlat, self.xy, lon, lat)
        else:
            return lon, lat
