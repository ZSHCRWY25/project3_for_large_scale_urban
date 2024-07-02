import gym
import gym_env
from pathlib import Path
from envs import env_init

world_name = 'gym_test_world.yaml'
# world_name = 'dynamic_obs_test.yaml'

env = gym.make('mdin-v1', world_name=world_name, drone_number=6)
env.reset()

for i in range(300):

    vel_list = env.ir_gym.cal_des_list()

    obs_list, reward_list, done_list, info_list, finish_list = env.step(vel_list)
    id_list=[id for id, done in enumerate(done_list) if done==True]
    
    for id in id_list: 
        env.reset_one(id)

    env.render()



