import xarray as xr


class SilcamSim:
    def __init__(self):
        """
        Init the silcam simulator.
        # Todo: add sensor charateristics (noise model)
        """

    def measure(self, ds, timestamp, pos):
        # x, y = self.lonlat2xy(lon=pos[0], lat=pos[1])
        x = pos[0]
        y = pos[1]
        # TODO: Proper 3d -> include zc=depth
        true_value = ds.isel(zc=0).biomass.interp(time=timestamp, xc=xr.DataArray(x), yc=xr.DataArray(y)).values
        # nan = np.isnan(true_value)
        # if any(nan):
        #    true_value[nan] = self.ds.isel(zc=0).biomass.sel(time=T0, xc=xr.DataArray(x[nan]), yc=xr.DataArray(y[nan]),
        #                                                method='nearest').values
        return true_value  # Todo: add noise
