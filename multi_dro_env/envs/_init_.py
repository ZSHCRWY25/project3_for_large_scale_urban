from gym.envs.registration import register

register(
    id='mdin-v1',
    entry_point='gym_env.envs:mdin',
)