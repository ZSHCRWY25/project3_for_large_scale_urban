import env_base
from math import sqrt, pi
from gym import spaces
from gym_env.envs.rvo_inter import rvo_inter
import numpy as np

class ir_gym(env_base):
    def __init__(self, world_name, neighbors_region=5, neighbors_num=10, vxmax = 1.5, vymax = 1.5, env_train=True, acceler = 0.5, **kwargs):
        super(ir_gym, self).__init__(world_name=world_name, **kwargs)

        # self.obs_mode = kwargs.get('obs_mode', 0)    # 0 drl_rvo, 1 drl_nrvo
        # self.reward_mode = kwargs.get('reward_mode', 0)

        self.radius_exp = kwargs.get('radius_exp', 0.2)#半径扩展参数

        self.env_train = env_train#环境是否处于训练模式

        self.nr = neighbors_region
        self.nm = neighbors_num#邻居区域和邻居数量

        self.rvo = rvo_inter(neighbors_region, neighbors_num, vxmax, vymax, acceler, env_train, self.radius_exp)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)#观测空间为5维
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)#动作空间为2维，范围在[-1, 1]之间
        
        self.reward_parameter = kwargs.get('reward_parameter', (0.2, 0.1, 0.1, 0.2, 0.2, 1, -20, 20)) #：奖励函数的参数
        self.acceler = acceler
        self.arrive_flag_cur = False#到达标志

        self.rvo_state_dim = 8#RVO（Reciprocal Velocity Obstacles）状态维度
        

    def cal_des_omni_list(self):#计算机器人的全向移动（omni-directional）的目标速度列表。
        des_vel_list = [robot.cal_des_vel_omni() for robot in self.robot_list]#调用其 cal_des_vel_omni() 方法以计算目标速度
        return des_vel_list


    def rvo_reward_list_cal(self, action_list, **kwargs):#计算机器人在执行一系列动作时的 RVO（Reciprocal Velocity Obstacles）奖励列表    
        ts = self.components['robots'].total_states() # robot_state_list, nei_state_list, obs_circular_list, obs_line_list#获取机器人的状态、邻居状态、圆形障碍物状态和线段障碍物状态

        rvo_reward_list = list(map(lambda robot_state, action: self.rvo_reward_cal(robot_state, ts[1], ts[2], ts[3], action, self.reward_parameter, **kwargs), ts[0], action_list))#对于每个动作，调用 rvo_reward_cal 方法计算奖励

        return rvo_reward_list
    
    def rvo_reward_cal(self, robot_state, nei_state_list, obs_cir_list, obs_line_list, action, reward_parameter=(0.2, 0.1, 0.1, 0.2, 0.2, 1, -10, 20), **kwargs):#计算机器人的 RVO 奖励
        
        vo_flag, min_exp_time, min_dis = self.rvo.config_vo_reward(robot_state, nei_state_list, obs_cir_list, obs_line_list, action, **kwargs)

        des_vel = np.round(np.squeeze(robot_state[-2:]), 2)
        
        p1, p2, p3, p4, p5, p6, p7, p8 = reward_parameter

        dis_des = sqrt((action[0] - des_vel[0] )**2 + (action[1] - des_vel[1])**2)
        max_dis_des = 3
        dis_des_reward = - dis_des / max_dis_des #  (0-1)
        exp_time_reward = - 0.2/(min_exp_time+0.2) # (0-1)
        
        # rvo reward    
        if vo_flag:
            rvo_reward = p2 + p3 * dis_des_reward + p4 * exp_time_reward
            
            if min_exp_time < 0.1:
                rvo_reward = p2 + p1 * p4 * exp_time_reward
        else:
            rvo_reward = p5 + p6 * dis_des_reward
        
        rvo_reward = np.round(rvo_reward, 2)

        return rvo_reward

    def obs_move_reward_list(self, action_list, **kwargs):#计算机器人执行一系列动作时的观测和奖励
        ts = self.components['robots'].total_states() # robot_state_list, nei_state_list, obs_circular_list, obs_line_list

        obs_reward_list = list(map(lambda robot, action: self.observation_reward(robot, ts[1], ts[2], ts[3], action, **kwargs), self.robot_list, action_list))

        obs_list = [l[0] for l in obs_reward_list]
        reward_list = [l[1] for l in obs_reward_list]
        done_list = [l[2] for l in obs_reward_list]
        info_list = [l[3] for l in obs_reward_list]
        #对于每个机器人，调用 observation_reward 方法计算观测和奖励。
#       返回观测列表、奖励列表、完成标志列表和其他信息列表。

        return obs_list, reward_list, done_list, info_list

    def observation_reward(self, robot, nei_state_list, obs_circular_list, obs_line_list, action, **kwargs):#算机器人的观测和奖励。
       # 获取机器人的状态、邻居状态、圆形障碍物状态、线段障碍物状态和动作。
       # 计算机器人的内部观测和外部观测。
       # 返回观测、奖励、完成标志和其他信息。

        robot_omni_state = robot.omni_state()
        des_vel = np.squeeze(robot.cal_des_vel_omni())
       
        done = False

        if robot.arrive() and not robot.arrive_flag:
            robot.arrive_flag = True
            arrive_reward_flag = True
        else:
            arrive_reward_flag = False

        obs_vo_list, vo_flag, min_exp_time, collision_flag = self.rvo.config_vo_inf(robot_omni_state, nei_state_list, obs_circular_list, obs_line_list, action, **kwargs)

        radian = robot.state[2]
        cur_vel = np.squeeze(robot.vel_omni)
        radius = robot.radius_collision* np.ones(1,)

        propri_obs = np.concatenate([ cur_vel, des_vel, radian, radius]) 
        
        if len(obs_vo_list) == 0:
            exter_obs = np.zeros((self.rvo_state_dim,))
        else:
            exter_obs = np.concatenate(obs_vo_list) # vo list
            
        observation = np.round(np.concatenate([propri_obs, exter_obs]), 2)

        # dis2goal = sqrt( robot.state[0:2] - robot.goal[0:2])
        mov_reward = self.mov_reward(collision_flag, arrive_reward_flag, self.reward_parameter, min_exp_time)

        reward = mov_reward

        done = True if collision_flag else False
        info = True if robot.arrive_flag else False
        
        return [observation, reward, done, info]

    def mov_reward(self, collision_flag, arrive_reward_flag, reward_parameter=(0.2, 0.1, 0.1, 0.2, 0.2, 1, -20, 15), min_exp_time=100, dis2goal=100):#计算机器人的移动奖励,根据碰撞标志、到达标志和其他参数计算奖励。

        p1, p2, p3, p4, p5, p6, p7, p8 = reward_parameter

        collision_reward = p7 if collision_flag else 0
        arrive_reward = p8 if arrive_reward_flag else 0
        time_reward = 0
        
        mov_reward = collision_reward + arrive_reward + time_reward

        return mov_reward

    def osc_reward(self, state_list):#避免机器人的振荡（oscillation）检查机器人状态列表中的角度变化，如果出现振荡则返回负奖励
        # to avoid oscillation
        dif_rad_list = []
        
        if len(state_list) < 3:#状态列表 state_list,三个以上
            return 0

        for i in range(len(state_list) - 1):#计算相邻状态之间的角度变化（差值）
            dif = ir_gym.wraptopi(state_list[i+1][2, 0] - state_list[i][2, 0])
            dif_rad_list.append(round(dif, 2))

        for j in range(len(dif_rad_list)-3):
            
            if dif_rad_list[j] * dif_rad_list[j+1] < -0.05 and dif_rad_list[j+1] * dif_rad_list[j+2] < -0.05 and dif_rad_list[j+2] * dif_rad_list[j+3] < -0.05:#如果连续三个状态的角度变化方向相反（例如从正转到负，再到正），则判断机器人出现了振荡
                print('osc', dif_rad_list[j], dif_rad_list[j+1], dif_rad_list[j+2], dif_rad_list[j+3])
                return -10
        return 0

    def observation(self, robot, nei_state_list, obs_circular_list, obs_line_list):#计算机器人的观测，基于其状态、邻居状态、圆形障碍物状态和线段障碍物状态

        robot_omni_state = robot.omni_state()#提取机器人的当前速度、期望速度、方向角和碰撞半径
        des_vel = np.squeeze(robot_omni_state[-2:])
        
        obs_vo_list, _, min_exp_time, _ = self.rvo.config_vo_inf(robot_omni_state, nei_state_list, obs_circular_list, obs_line_list)#如果存在速度障碍物（VO），还计算外部观测
    
        cur_vel = np.squeeze(robot.vel_omni)
        radian = robot.state[2]
        radius = robot.radius_collision* np.ones(1,)

        if len(obs_vo_list) == 0:
            exter_obs = np.zeros((self.rvo_state_dim,))
        else:
            exter_obs = np.concatenate(obs_vo_list) # vo list

        
        propri_obs = np.concatenate([ cur_vel, des_vel, radian, radius]) 
        observation = np.round(np.concatenate([propri_obs, exter_obs]), 2)#将内部和外部观测连接成最终的观测向量。

        return observation

    def env_reset(self, reset_mode=1, **kwargs):#重置环境，包括机器人
        
        self.components['robots'].robots_reset(reset_mode, **kwargs)
        ts = self.components['robots'].total_states()
        obs_list = list(map(lambda robot: self.observation(robot, ts[1], ts[2], ts[3]), self.robot_list))
#调用 robots_reset 方法以重置机器人状态。
#根据更新后的状态计算所有机器人的观测
        return obs_list

    def env_reset_one(self, id):#重置环境中的特定机器人，在多智能体场景中逐个重置机器人
        self.robot_reset(id)

    def env_observation(self):#计算环境中所有机器人的观测，类似于 env_reset，但不重置环境或机器人状态
        ts = self.components['robots'].total_states()
        obs_list = list(map(lambda robot: self.observation(robot, ts[1], ts[2], ts[3]), self.robot_list))

        return obs_list

    @staticmethod
    def wraptopi(theta):

        if theta > pi:
            theta = theta - 2*pi
        
        if theta < -pi:
            theta = theta + 2*pi

        return theta
    