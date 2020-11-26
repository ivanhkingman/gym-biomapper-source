import gym
from gym import error, spaces, utils
from gym.utils import seeding

import matplotlib.pyplot as plt
import cmocean

import datetime
import numpy as np
import xarray as xr
import pyproj

from gym_biomapping.envs.silcam_sim import SilcamSim
from gym_biomapping.envs.auv_sim import AUVsim


class BioMapping(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dt=60, lon0=10.4, lat0=63.44, data_src='data/bio2d_v2_samples_TrF_2018.04.27.nc'):
        super(BioMapping, self).__init__()
        assert (np.shape(lon0) == np.shape(lat0))
        assert (dt > 0)
        # dt = pd.tseries.offsets.DateOffset(seconds=dt) # sek
        # self.times = pd.date_range('2018-04-27 10:00:00', freq=dt, periods=N, tz='UTC')
        # self.data = np.zeros((N,n_auvs,4))
        self.dt = dt
        self.ds = xr.open_dataset(data_src)
        self.xy = pyproj.Proj(proj=self.ds.proj_xy)
        self.lonlat = pyproj.Proj(proj=self.ds.proj_lonlat)
        self.t0 = self.ds.time.values[0]
        # self.t0 = datetime.datetime(year=2018, month=4, day=27, hour=10, minute=0,
        #                            tzinfo=datetime.timezone.utc).timestamp()
        self.t = self.t0

        n_auvs = np.size(lon0)

        self.silcam_sim = SilcamSim()
        x0, y0 = self.lonlat2xy(lon0, lat0)
        self.auv_sim = AUVsim(lon0=x0, lat0=y0, speed=1.0, dt=dt)
        # Lats = np.linspace(63.4294, 63.4800, 50)
        # Lons = np.linspace(10.3329, 10.4461, 50)
        # TODO: Determine appropriate action space.
        # Action space is just a waypoint anywhere in the world
        low = np.array([np.min(self.ds.xc), np.min(self.ds.yc), np.min(self.ds.zc)], dtype=np.float32)
        high = np.array([np.max(self.ds.xc), np.max(self.ds.yc), np.max(self.ds.zc)], dtype=np.float32)
        np.repeat(np.array([2, 3, 4])[:, np.newaxis], n_auvs, axis=1)
        pos_space = spaces.Box(low=np.repeat(low[:, np.newaxis], n_auvs, axis=1),
                               high=np.repeat(high[:, np.newaxis], n_auvs, axis=1),
                               shape=(3, n_auvs),
                               dtype=np.float32)
        time_space = spaces.Box(low=self.ds.time.values.min(),
                                high=self.ds.time.values.max(),
                                shape=(1, n_auvs))
        # TODO: Determine if observations should be normalised
        env_space = spaces.Box(low=self.ds.biomass.values.min(), high=self.ds.biomass.values.max(),
                               shape=(self.ds.biomass.xc.size, self.ds.biomass.yc.size), dtype=np.float32)

        self.action_space = pos_space
        self.observation_space = spaces.Dict({"pos": pos_space, "time": time_space, "env": env_space})

        self.figure_initialised = False

    def step(self, action):
        self.t += self.dt
        pos = self.auv_sim.step(action)
        obs = self.silcam_sim.measure(self.ds, self.t, pos)
        env_state = self.ds.isel(zc=0).biomass.interp(time=self.t).T.values
        state = {
            "pos": pos,
            "time": np.ones(obs.shape) * self.t,
            "env": env_state
        }
        reward = np.linalg.norm(obs, axis=0)
        done = False
        info = {'sim_time': self.t - self.t0}
        return state, reward, done, info

    def reset(self):
        self.t = self.t0
        pos = self.auv_sim.reset()
        obs = self.silcam_sim.measure(self.ds, self.t, pos)
        return np.array([pos[0], pos[1], pos[2], obs, np.ones(obs.shape) * self.t]).T

    def render(self, mode='human'):
        if not self.figure_initialised:
            self.fig, self.ax = plt.subplots()
            self.figure_initialised = True

        self.ax.clear()
        self.ax.pcolormesh(self.ds.xc, self.ds.yc,
                           self.ds.isel(zc=0).biomass.interp(time=self.t).T,
                           cmap=cmocean.cm.deep,
                           shading='auto')
        self.ax.plot(self.auv_sim.pos[0], self.auv_sim.pos[1], 'ro')
        self.ax.set_title("Simulation duration: " + str(self.t - self.t0) + "s")
        self.fig.canvas.draw()

    def close(self):
        return

    def xy2lonlat(self, x, y):
        # Convert from x-y coordinates to lon-lat
        # (this function works on both scalars and arrays)
        return pyproj.transform(self.xy, self.lonlat, x, y)

    def lonlat2xy(self, lon, lat):
        # Convert from lon-lat to x-y coordinates
        # (this function works on both scalars and arrays)
        return pyproj.transform(self.lonlat, self.xy, lon, lat)
