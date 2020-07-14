from gym.envs.registration import register

register(
    id='biomapping-v0',
    entry_point='gym_biomapping.envs:BioMappingEnv',
)
