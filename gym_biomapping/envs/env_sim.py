from gym import error, spaces, utils
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import pyproj
from gym_biomapping.envs.silcam_sim import SilcamSim
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

DATA_DIR = Path(__file__).parent.joinpath('data')


class EnvSim:
    def __init__(self, data_file='bio2d_v2_samples_TrF_2018.04.27.nc', exact_indexed=True, static=False,
                 generate_data=False, n_auvs=1):
        self.static = static
        data_path = DATA_DIR.joinpath(data_file)
        if not data_path.is_file():
            data_path = data_file
        self.ds = xr.open_dataset(data_path)
        self.xy = pyproj.Proj(proj=self.ds.proj_xy)
        self.lonlat = pyproj.Proj(proj=self.ds.proj_lonlat)
        self.exact_indexed = exact_indexed
        self.t0 = self.ds.time.values[0]

        self.silcam_sim = SilcamSim(exact_indexed=exact_indexed)

        # Pos space is a set of points anywhere in the env, one point for each AUV
        if exact_indexed:
            self.pos_space = spaces.MultiDiscrete([self.ds.xc.size, self.ds.yc.size, self.ds.zc.size])
        else:
            low = np.array([np.min(self.ds.xc), np.min(self.ds.yc), np.min(self.ds.zc)], dtype=np.float32)
            high = np.array([np.max(self.ds.xc), np.max(self.ds.yc), np.max(self.ds.zc)], dtype=np.float32)
            self.pos_space = spaces.Box(low=np.repeat(low[np.newaxis, :], n_auvs, axis=0),
                                        high=np.repeat(high[np.newaxis, :], n_auvs, axis=0),
                                        shape=(n_auvs, 3))
        self.time_space = spaces.Box(low=self.ds.time.values.min(),
                                     high=self.ds.time.values.max(),
                                     shape=(1, 0))
        # TODO: Determine if observations should be normalised
        self.env_space = spaces.Box(low=self.ds.biomass.values.min(), high=self.ds.biomass.values.max(),
                                    shape=(self.ds.biomass.xc.size, self.ds.biomass.yc.size))

        self.figure_initialised = False
        self.generate_data = generate_data
        if generate_data:
            np.random.seed(0)  # All np.random calls from here on will be reproducible.
            # Every time reset() is called a new random environment will be drawn, but the
            # sequence of environments will be the same for each gym instance.
            kernel = Matern(length_scale=300, nu=1.5)
            self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
        self.reset()

    def step(self, pos, t):

        if self.static:
            measure = self.silcam_sim.measure(self.ds, self.t0, pos)
            self.env_state = self.ds.isel(zc=0).biomass.interp(time=self.t0).values
        else:
            measure = self.silcam_sim.measure(self.ds, self.t0+t, pos)
            self.env_state = self.ds.isel(zc=0).biomass.interp(time=self.t0+t).values

        return self.env_state, measure

    def reset(self):
        self.t = self.t0
        self.steps_taken = 0
        if self.generate_data:
            xc = self.ds.xc.values
            yc = self.ds.yc.values
            x1, x2 = np.broadcast_arrays(xc.reshape(-1, 1), yc.reshape(1, -1))
            xy = np.vstack((x1.flatten(), x2.flatten())).T
            y_mean = 25 * self.gp.sample_y(xy, random_state=None) + 50
            self.ds['biomass'][dict(zc=0)].loc[dict(time=self.t0)] = y_mean.reshape(len(xc), len(yc))
        self.env_state = self.ds.isel(zc=0).sel(time=self.t0).biomass.values

        return self.env_state

    def render(self, ax):
        if self.exact_indexed:
            ax.pcolormesh(self.env_state.T,
                          cmap=cmocean.cm.deep,
                          shading='auto')
        else:
            ax.pcolormesh(self.ds.xc, self.ds.yc,
                          self.env_state.T,
                          cmap=cmocean.cm.deep,
                          shading='auto')
