'''
:@Author: 刘玉璞
:@Date: 2024/6/24 16:44:00
:@LastEditors: 刘玉璞
:@LastEditTime: 2024/7/2 16:44:00
:Description: 
'''
import gym
from envs.env.ir_gym import ir_gym

class mdin(gym.Env):
    def __init__(self, world_name=None, neighbors_region=5, neighbors_num=10, **kwargs):
        
        self.ir_gym = ir_gym(world_name, neighbors_region, neighbors_num, **kwargs)
        
        self.observation_space = self.ir_gym.observation_space
        self.action_space = self.ir_gym.action_space
        
        self.neighbors_region = neighbors_region


    # def step(self, action, vel_type='omni', stop=True, **kwargs):
    def step(self, action, stop=True, **kwargs):

        if not isinstance(action, list):
            action = [action]

        rvo_reward_list = self.ir_gym.rvo_reward_list_cal(action)
        self.ir_gym.drone_step(action, stop=stop)
        obs_list, mov_reward, done_list, info_list, finish_list = self.ir_gym.obs_move_reward_list(action, **kwargs)

        reward_list = [x+y for x, y in zip(rvo_reward_list, mov_reward)]
        
        return obs_list, reward_list, done_list, info_list, finish_list

    def render(self, mode = 'human', save=False, path=None, i = 0, **kwargs):
        self.ir_gym.render(0.01, **kwargs)

        if save:
            self.ir_gym.save_fig(path, i) 

    def reset(self):
        # mode = kwargs.get('mode', 0)
        return self.ir_gym.env_reset()

    def reset_one(self, id):
        self.ir_gym.components['drones'].drone_reset(id)

    def show(self):
        self.ir_gym.show()