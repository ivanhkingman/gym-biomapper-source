import setuptools

setuptools.setup(name='gym-biomapping',
      version='0.0.1',
      author="AndreasVaage",
      author_email="andreas.vage@ntnu.no",
      description="Simulator for adaptive sampling of biomass in the ocean",
      packages=setuptools.find_packages(),
      include_package_data=True,
      package_data={'':['*.nc'],},
      install_requires=['gym', 'numpy', 'pyproj', 'xarray', 'netcdf4', 'matplotlib', 'cmocean', 'datetime']
)
