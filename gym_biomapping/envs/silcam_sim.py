import xarray as xr
import pyproj


class SilcamSim:
    def __init__(self, src):
        """
        Init the silcam simulator.
        :param src: path to simulation data
        :param sinmoddata: used in sim mode
        """
        self.ds = xr.open_dataset(src)
        self.xy = pyproj.Proj(proj=self.ds.proj_xy)
        self.lonlat = pyproj.Proj(proj=self.ds.proj_lonlat)

    def xy2lonlat(self, x, y):
        # Convert from x-y coordinates to lon-lat
        # (this function works on both scalars and arrays)
        return pyproj.transform(self.xy, self.lonlat, x, y)

    def lonlat2xy(self, lon, lat):
        # Convert from lon-lat to x-y coordinates
        # (this function works on both scalars and arrays)
        return pyproj.transform(self.lonlat, self.xy, lon, lat)

    def measure(self, timestamp, pos):
        # x, y = self.lonlat2xy(lon=pos[0], lat=pos[1])
        x = pos[0]
        y = pos[1]
        # TODO: Proper 3d -> include zc=depth
        true_value = self.ds.isel(zc=0).biomass.interp(time=timestamp, xc=xr.DataArray(x), yc=xr.DataArray(y)).values
        # nan = np.isnan(true_value)
        # if any(nan):
        #    true_value[nan] = self.ds.isel(zc=0).biomass.sel(time=T0, xc=xr.DataArray(x[nan]), yc=xr.DataArray(y[nan]),
        #                                                method='nearest').values
        return true_value  # Todo: add noise