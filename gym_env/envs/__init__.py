from gym.envs.registration import register

register(
    id='mdin-v1',
    entry_point='envs.env:mdin',
)