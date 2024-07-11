'''
:@Author: 刘玉璞
:@Date: 2024/6/24 16:43:57
:@LastEditors: 刘玉璞
:@LastEditTime: 2024/7/2 16:43:57
:Description: 
'''
from envs.env.env_base import env_base
from math import sqrt, pi, acos, degrees
from gym import spaces
from envs.vel_obs.rvo_inter import rvo_inter
import numpy as np

class ir_gym(env_base):
    def __init__(self, world_name, neighbors_region=5, neighbors_num=10, vxmax = 2, vymax = 2, vzmax = 2, env_train=True, acceler = 0.5, **kwargs):
        super(ir_gym, self).__init__(world_name=world_name, **kwargs)#（改完）

        # self.obs_mode = kwargs.get('obs_mode', 0)    # 0 drl_rvo, 1 drl_nrvo
        # self.reward_mode = kwargs.get('reward_mode', 0)

        self.env_train = env_train#环境是否处于训练模式

        self.nr = neighbors_region
        self.nm = neighbors_num#邻居区域和邻居数量

        self.rvo = rvo_inter(neighbors_region, neighbors_num, vxmax, vymax,vzmax, acceler, env_train)
        #def __init__(self, neighbor_region=5, neighbor_num=10, vxmax=1.5, vymax=1.5, vzmax = 1.5, acceler = 0.3, env_train=True, exp_radius=0.2, ctime_threshold=5, delta_t = 1):
        #super(rvo_inter, self).__init__(neighbor_region, vxmax, vymax, vzmax, acceler)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)#观测空间为5维
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)#动作空间为3维，范围在[-1, 1]之间
        
        self.reward_parameter = kwargs.get('reward_parameter', (0.2, 0.1, 0.1, 0.2, 0.2, 1, -10, 20, 20, -2)) #：奖励函数的参数
        self.acceler = acceler
        self.arrive_flag_cur = False#到达标志

        self.rvo_state_dim = 9#RVO（Reciprocal Velocity Obstacles）状态维度
        

    def cal_des_list(self):#计算所有无人机的目标速度列表。(改完)
        des_vel_list = [drone.cal_des_vel() for drone in self.drone_list]
        return des_vel_list


    def rvo_reward_list_cal(self, action_list, **kwargs):#计算机器人在执行一系列动作时的 RVO（Reciprocal Velocity Obstacles）奖励列表   （改完了，在mrnav中调用） 
        drone_state_list = self.components['drones'].total_states() # #所有无人机状态,这个函数在env_drons里面
        rvo_reward_list = [] 
        for i, (drone_state, action) in enumerate(zip(drone_state_list, action_list)):  
        # 排除当前无人机状态，获取其他无人机的状态列表  
            other_drone_states = [s for j, s in enumerate(drone_state_list) if j != i]  
            rvo_reward = self.rvo_reward_cal(drone_state, other_drone_states, action, self.reward_parameter, **kwargs)#所有无人机选择的动作进行打分  
            rvo_reward_list.append(rvo_reward)  
        return rvo_reward_list
    
    def rvo_reward_cal(self, drone_state, other_drone_states, action, reward_parameter=(0.2, 0.1, 0.1, 0.2, 0.2, 1, -10, 20, 20, -2), **kwargs):#计算无人机的 RVO 奖励(0.2, 0.1, 0.1, 0.2, 0.2, 1, -10, 20, 20, -2)
        
        vo_flag, min_exp_time, min_dis = self.rvo.config_vo_reward(drone_state, other_drone_states, self.building_list, action, **kwargs)#self, drone_state, other_drone_state_list,  building_list, action=np.zeros((2,)), **kwargs

        des_vel = np.round(np.squeeze(drone_state[-3:]), 1)
         
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = reward_parameter

        dis_des = sqrt((action[0] - des_vel[0] )**2 + (action[1] - des_vel[1])**2+(action[2] - des_vel[2] )**2)
        max_dis_des = 3
        dis_des_reward = - dis_des / max_dis_des #  (0-1)
        #exp_time_reward = - 0.2/(min_exp_time+0.2) # (0-1)
        min_dis_reward = - 0.2/(min_dis + 0.2)
        # rvo reward    
        if vo_flag:
            rvo_reward = p2 + p3 * dis_des_reward + p4 * min_dis_reward #exp_time_reward
            
            if min_exp_time < 0.1:
                rvo_reward = p2 + p1 * p4 * min_dis_reward #exp_time_reward
        else:
            rvo_reward = p5 + p6 * dis_des_reward
        
        rvo_reward = np.round(rvo_reward, 2)

        return rvo_reward

    def obs_move_reward_list(self, action_list, **kwargs):#计算机器人执行一系列动作时的观测和奖励
        drone_state_list = self.components['drones'].total_states() 
        obs_reward_list = []
        for i, (drone, action) in enumerate(zip(self.drone_list , action_list)):  
        # 排除当前无人机状态，获取其他无人机的状态列表  
            other_drone_state_list = [s for j, s in enumerate(drone_state_list ) if j != i]
            obs, reward, done, info, finish = self.observation_reward(drone, other_drone_state_list, action)
            #obs, reward, done, info = 
            obs_reward_list.append((obs, reward, done, info, finish))

        observation_list = [l[0] for l in obs_reward_list]
        reward_list = [l[1] for l in obs_reward_list]
        done_list = [l[2] for l in obs_reward_list]
        info_list = [l[3] for l in obs_reward_list]
        finish_list = [l[4] for l in obs_reward_list]
        #计算观测和奖励。
#       返回观测列表、奖励列表、完成标志列表和其他信息列表。

        return observation_list, reward_list, done_list, info_list, finish_list 

    def observation_reward(self, drone, odro_state_list,action):#算无人机的观测和奖励。

       # 计算无人机的内部观测和外部观测。
       # 返回观测、奖励、完成标志和其他信息。
        drone_state = drone.dronestate()
        des_vel = np.squeeze(drone.cal_des_vel())
        destination_arrive_reward_flag = False
        done = False
        
        if drone.arrive(drone.state, drone.current_des) and not drone.arrive_flag:##到途经航路点奖励5
            drone.arrive_flag = True
            arrive_reward_flag = True
        else:
            arrive_reward_flag = False

        if drone.arrive_flag == True:

            if drone.destination_arrive() and not drone.destination_arrive_flag:#到最终目的地奖励
                drone.destination_arrive_flag = True
                destination_arrive_reward_flag = True
            else:
                destination_arrive_reward_flag = False

        deviation = drone.Deviation_from_route()
        #drone_state, drone_state_list, building_list, action
        obs_vo_list, vo_flag, min_dis, collision_flag, obs_building_list = self.rvo.config_vo_inf(drone_state, odro_state_list, self.building_list, action)
        #obs_vo_list_nm, vo_flag, min_exp_time, collision_flag, obs_building_list
        cur_vel = np.squeeze(drone.vel)
        radius = drone.radius_collision* np.ones(1,)

        propri_obs = np.concatenate([cur_vel, des_vel, radius]) ##内部观测
        
        if len(obs_vo_list) == 0:
            exter_obs_vo = np.zeros((self.rvo_state_dim,))
        else:
            exter_obs_vo = np.concatenate(obs_vo_list) # vo list外部观测

        if len(obs_building_list) == 0:
            exter_obs_building = np.zeros((self.rvo_state_dim,))
        else:
            exter_obs_building = np.concatenate(obs_building_list) # vo list外部观测
            
        observation = np.round(np.concatenate([propri_obs, exter_obs_vo, exter_obs_building]), 2)##链接

        # dis2goal = sqrt( robot.state[0:2] - robot.goal[0:2
        mov_reward = self.mov_reward(collision_flag, arrive_reward_flag, destination_arrive_reward_flag, deviation, self.reward_parameter, min_dis)

        reward = mov_reward

        done = True if collision_flag else False
        info = True if drone.arrive_flag else False
        finish = True if drone.destination_arrive_flag else False
        #done = True if drone.destination_arrive_flag else False
        #info也可以处理更多信息：info = {'arrive_flag': drone.arrive_flag, 'destination_arrive_flag': drone.destination_arrive_flag}  
        
        return observation, reward, done, info, finish##[内外观测（自身+其他无人机速度障碍+冲突建筑物），移动奖励（碰撞+到点+终点），碰撞标准，到点标志 ，到终点标志]

    def mov_reward(self, collision_flag, arrive_reward_flag, destination_arrive_reward_flag, deviation, reward_parameter=(0.2, 0.1, 0.1, 0.2, 0.2, 1, -10, 20, 20, -2), min_exp_time=100, dis2goal=100):#计算机器人的移动奖励,根据碰撞标志、到达标志和其他参数计算奖励。

        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = reward_parameter

        collision_reward = p7 if collision_flag else 0
        arrive_reward = p8 if arrive_reward_flag else 0
        destination_arrive_reward = p9 if destination_arrive_reward_flag else 0
        time_reward = 0
        #time_reward = -p6 * (actual_time - min_exp_time) if actual_time is not None and actual_time > min_exp_time else 0  
        deviation_reward = deviation * p10
        mov_reward = collision_reward + arrive_reward + time_reward + destination_arrive_reward+deviation_reward

        return mov_reward

    def osc_reward(self, state_list):#避免轨迹振荡（oscillation）检查状态列表中的角度变化，如果出现振荡则返回负奖励
        # to avoid oscillation
        dif_rad_list = []
        
        if len(state_list) < 3:#状态列表 state_list,三个以上
            return 0

        for i in range(1,len(state_list) - 1):#计算相邻状态之间的角度变化（差值）
            angle1 = self.calculate_angle_between_vectors(state_list[i+1][3:6], state_list[i][3:6])#[x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
            angle2 = self.calculate_angle_between_vectors(state_list[i][3:6], state_list[i-1][3:6])
            dif = self.wraptopi(angle1 - angle2)
            dif_rad_list.append(round(dif, 2))
            
        for j in range(len(dif_rad_list) - 3):  
            # 检测连续三个角度变化的方向是否相反   
            if (dif_rad_list[j] > 0 and dif_rad_list[j+1] < 0 and dif_rad_list[j+2] > 0) or \
                (dif_rad_list[j] < 0 and dif_rad_list[j+1] > 0 and dif_rad_list[j+2] < 0):
                print('osc', dif_rad_list[j], dif_rad_list[j+1], dif_rad_list[j+2], dif_rad_list[j+3])  
                return -10  
        return 0

    def observation(self, drone, other_drone_state_list):#计算观测
        drone_state = drone.dronestate() #提取当前位置、速度、大小、优先级、期望速度
        des_vel = np.squeeze(drone_state[-3:])# state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
        
        obs_vo_list, _, min_dis, _ , obs_building_list= self.rvo.config_vo_inf(drone_state, other_drone_state_list, self.building_list)#如果存在速度障碍物（VO），还计算外部观测
        cur_vel = np.squeeze(drone.vel)
        radius = drone.radius_collision* np.ones(1,)

        if len(obs_vo_list) == 0:
            exter_obs = np.zeros((self.rvo_state_dim,))
        else:
            exter_obs = np.concatenate(obs_vo_list) # vo list外部观测变量

        if len(obs_building_list) == 0:
            exter_obs_building = np.zeros((self.rvo_state_dim,))
        else:
            exter_obs_building = np.concatenate(obs_building_list) # vo list外部观测

        
        propri_obs = np.concatenate([ cur_vel, des_vel, radius]) #内部变量
        observation = np.round(np.concatenate([propri_obs, exter_obs, exter_obs_building]), 2)#将内部和外部观测连接成最终的观测向量。


        return observation

    def env_reset(self):#重置环境

        self.components['drones'].drones_reset()
        drone_state_list = self.components['drones'].total_states()# state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
        obs_list = list(map(lambda drone: self.observation(drone, drone_state_list ), self.drone_list))
    #调用reset 方法以重置无人机状态。
    #根据更新后的状态计算所有无人机的观测
        return obs_list

    def env_reset_one(self, id):#重置环境中的特定无人机，在多智能体场景中逐个重置无人机
        self.drone_reset(id)

    def env_observation(self):#计算环境中所有无人机的观测，类似于 env_reset，但不重置环境或无人机状态
        drone_list = self.components['drones'].Drone_list()
        drone_state_list = self.components['drones'].total_states() 
        observation_list = []
        for i, (drone) in enumerate(zip(drone_list)):  
        # 排除当前无人机状态，获取其他无人机的状态列表  
            other_drone_state = [s for j, s in enumerate(drone_state_list ) if j != i]
            observation = self.observation(drone, other_drone_state,self.building_list)
            observation_list.append((observation))

        return observation_list
    
    def render(self, time=0.05, **kwargs):

        if self.plot:
            self.world_plot.clear_plot_elements()
            self.world_plot.draw_drones(**kwargs)
            self.world_plot.pause(time)
            
        self.time = self.time + time

    @staticmethod
    def wraptopi(theta):

        if theta > pi:
            theta = theta - 2*pi
        
        if theta < -pi:
            theta = theta + 2*pi

        return theta
    
    @staticmethod
    def calculate_angle_between_vectors(A, B):

        # 计算向量A和B的模  
        mag_A = sqrt(A[0]**2 + A[1]**2 + A[2]**2)  
        mag_B = sqrt(B[0]**2 + B[1]**2 + B[2]**2)  
      
        # 计算向量A和B的点积  
        dot_product = A[0] * B[0] + A[1] * B[1] + A[2] * B[2]  
      
        # 计算夹角的余弦值  
        cos_theta = dot_product / (mag_A * mag_B)  
      
        # 计算夹角（以弧度为单位）  
        theta_radians = acos(cos_theta)  
      
      
        return theta_radians
    
    