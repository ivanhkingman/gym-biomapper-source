from gym import error, spaces, utils
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import pyproj
from gym_biomapping.envs.silcam_sim import SilcamSim

DATA_DIR = Path(__file__).parent.joinpath('data')


class EnvSim:
    def __init__(self, dt=60, data_file='bio2d_v2_samples_TrF_2018.04.27.nc', static=False, n_auvs=1):
        assert (dt > 0)
        self.static = static
        self.dt = dt
        data_path = DATA_DIR.joinpath(data_file)
        if not data_path.is_file():
            data_path = data_file
        self.ds = xr.open_dataset(data_path)
        self.xy = pyproj.Proj(proj=self.ds.proj_xy)
        self.lonlat = pyproj.Proj(proj=self.ds.proj_lonlat)
        self.t0 = self.ds.time.values[0]

        self.silcam_sim = SilcamSim()

        # Action space is a set of points anywhere in the env, one point for each AUV
        low = np.array([np.min(self.ds.xc), np.min(self.ds.yc), np.min(self.ds.zc)], dtype=np.float32)
        high = np.array([np.max(self.ds.xc), np.max(self.ds.yc), np.max(self.ds.zc)], dtype=np.float32)
        np.repeat(np.array([2, 3, 4])[:, np.newaxis], n_auvs, axis=1)
        self.pos_space = spaces.Box(low=np.repeat(low[:, np.newaxis], n_auvs, axis=1),
                                    high=np.repeat(high[:, np.newaxis], n_auvs, axis=1),
                                    shape=(3, n_auvs),
                                    dtype=np.float32)
        # TODO: Determine if observations should be normalised
        self.env_space = spaces.Box(low=self.ds.biomass.values.min(), high=self.ds.biomass.values.max(),
                                    shape=(self.ds.biomass.xc.size, self.ds.biomass.yc.size), dtype=np.float32)

        self.figure_initialised = False
        self.reset()

    def step(self, pos):
        self.t += self.dt

        if self.static:
            obs = self.silcam_sim.measure(self.ds, self.t0, pos)
            self.env_state = self.ds.isel(zc=0).biomass.interp(time=self.t0).values
        else:
            obs = self.silcam_sim.measure(self.ds, self.t, pos)
            self.env_state = self.ds.isel(zc=0).biomass.interp(time=self.t).values

        reward = np.linalg.norm(obs, axis=0)
        done = False
        info = {'sim_time': self.t - self.t0}
        return self.env_state, reward, done, info

    def reset(self):
        self.t = self.t0
        self.env_state = self.ds.isel(zc=0).biomass.interp(time=self.t0).values

        return self.env_state

    def render(self, ax):
        ax.set_title("Simulation duration: " + str(self.t - self.t0) + "s")
        ax.pcolormesh(self.ds.xc, self.ds.yc,
                      self.env_state.T,
                      cmap=cmocean.cm.deep,
                      shading='auto')
