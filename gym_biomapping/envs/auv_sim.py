import pyproj
import numpy as np

class AUVsim():
    
    #metadata = {
    #    'render.modes': ['human', 'rgb_array'],
    #    'plot.frames_per_second': 50
    #}
    
    def __init__(self, lon0 = 0.0, lat0 = 0.0, speed=1.0, dt = 1.0, xy=None, lonlat=None):
        assert np.size(lon0) == np.size(lat0), "lat0 and lon0 must be same size"
        self.nauvs = np.size(lon0)
        if np.size(speed) != 1:
            assert np.size(speed) == self.nauvs, "Speed must be the same size at lon0"
            assert all(speed > 0), "All auv speeds must be positive"
        else:
            assert speed > 0, "Speed must be positive"
        assert(dt > 0), "Timestep must be positive scalar"
        self.lon0 = lon0
        self.lat0 = lat0
        self.speed = speed                    # m/s
        self.dt = dt                          # s
        self.xy = xy
        self.lonlat = lonlat
        self.reset()
        
    def step(self, action):
        """
        Do one simulation step
        :param action  is 3xN matrix, where N is the number of AUVs
        action[0][5] returns x value of the goal position for auv number 6.
        :return state, current position of the AUVs, same structure as action.
        """
    
        #goal = np.zeros((3, self.nauvs))
        #goal[0], goal[1] = self.lonlat2xy(action[0], action[1])

        dv = action - self.pos
        dist = np.linalg.norm(dv, axis=0)
        
        at_goal = dist < self.speed*self.dt
        self.pos[:, at_goal] = action[:, at_goal]
        self.pos[:, ~at_goal] += (dv[:, ~at_goal]/dist[~at_goal])*self.speed*self.dt
        
        #self.heading = np.arctan2(dv[1], dv[0])

        #lon, lat = self.xy2lonlat(self.pos[0], self.pos[1])
        #return np.array([lon, lat, self.pos[2]])
        return self.pos
        
    def reset(self):
        self.pos = np.zeros((3, self.nauvs))
        self.pos[0] = self.lon0
        self.pos[1] = self.lat0
        #self.pos[0], self.pos[1] = self.lonlat2xy(self.lon0, self.lat0)
        #return np.array([self.lon0, self.lat0, self.pos[2]])
        return self.pos
 
    def render(self, mode='human', close=False):
        pass
    
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
