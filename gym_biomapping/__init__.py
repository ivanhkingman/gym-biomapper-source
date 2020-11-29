from gym.envs.registration import register

register(
    id='perfect_info-v0',
    entry_point='gym_biomapping.envs:BioMapping',
    kwargs={'dt': 60,
            'lon0': 10.4,
            'lat0': 63.44,
            'data_file': 'bio2d_v2_samples_TrF_2018.04.27.nc'}
)
