import gym
from gym import error, spaces, utils
from gym.utils import seeding

import ipywidgets as widgets
import matplotlib.pyplot as plt
import cmocean

import datetime
import numpy as np
import pandas as pd
import xarray as xr
import sys

sys.path.append('/home/andreas/ailaron/SimplePlanner')
from silcamsim import SilcamSim
from auv_sim import AUVsim

class BioMapping(gym.Env):  
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dt = 60, lon0=10.4, lat0=63.44, every_n = 10):
        #dt = pd.tseries.offsets.DateOffset(seconds=dt) # sek
        #self.times = pd.date_range('2018-04-27 10:00:00', freq=dt, periods=N, tz='UTC')
        #self.data = np.zeros((N,n_auvs,4))
        self.dt = dt
        self.t0 = datetime.datetime(year=2018, month=4, day=27, hour=10, minute=0,
                                    tzinfo=datetime.timezone.utc).timestamp()
        self.t = self.t0

        src = '/home/andreas/ailaron/Ailaron-WP3/data/bio2d_v2_samples_TrF_2018.04.27.nc'
        self.silcam_sim = SilcamSim(src=src)
        x0, y0 = self.silcam_sim.lonlat2xy(lon0, lat0)
        self.auv_sim = AUVsim(lon0 = x0, lat0 = y0, speed=1.0, dt = dt)
        #Lats = np.linspace(63.4294, 63.4800, 50)
        #Lons = np.linspace(10.3329, 10.4461, 50)
        self.env_size = [[np.min(self.silcam_sim.ds.xc),
                          np.max(self.silcam_sim.ds.xc)],
                         [np.min(self.silcam_sim.ds.yc),
                          np.max(self.silcam_sim.ds.yc)]]
        
        self.figure_initialised=False

    def step(self, action):
        self.t += self.dt
        pos = self.auv_sim.step(action)
        self.obs = self.silcam_sim.measure(self.t, pos)
        state = np.array([pos[0], pos[1], pos[2], self.obs, np.ones(self.obs.shape)*self.t]).T
        reward = 0
        done = False
        info = {'sim_time': self.t-self.t0}
        return state, reward, done, info
    
    def reset(self):
        self.t = self.t0
        pos = self.auv_sim.reset()
        self.obs = self.silcam_sim.measure(self.t, pos)
        return np.array([pos[0], pos[1], pos[2], self.obs, np.ones(self.obs.shape)*self.t]).T
    
    def render(self, mode='human', close=False):
        if not self.figure_initialised:
            self.fig, self.ax = plt.subplots()
            self.figure_initialised=True
    
        self.ax.clear()
        self.ax.pcolormesh(self.silcam_sim.ds.xc, self.silcam_sim.ds.yc,
                           self.silcam_sim.ds.isel(zc=0).biomass.interp(time=self.t).T,
                           cmap=cmocean.cm.deep)
        self.ax.plot(self.auv_sim.pos[0], self.auv_sim.pos[1], 'ro')
        self.ax.set_title("Simulation duration: "+str(self.t-self.t0)+"s")
        self.fig.canvas.draw()