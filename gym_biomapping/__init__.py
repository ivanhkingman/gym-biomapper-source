from gym.envs.registration import register

register(
    id='perfect_info-v0',
    entry_point='gym_biomapping.envs:BioMapping',
    kwargs={'dt': 60,
            'pos0': None,
            'data_file': 'bio2d_v2_samples_TrF_2018.04.27.nc',
            'static': False,
            'output': 'dict',
            'n_auvs': 1}
)

register(
    id='perfect_info_atari-v0',
    entry_point='gym_biomapping.envs:AtariBioMapping',
    kwargs={'pos0': None,
            'data_file': 'bio2d_v2_samples_TrF_2018.04.27.nc',
            'static': False,
            'n_auvs': 1}
)
