'''
:@Author: 刘玉璞
:@Date: 2024/6/12 11:00:18
:@LastEditors: 刘玉璞
:@LastEditTime: 2024/6/12 11:00:18
:Description: 
'''
from gym_env.envs.reciprocal_vel_obs import reciprocal_vel_obs
from math import sqrt, atan2, asin, sin, pi, cos, inf
import numpy as np
from vel_obs3D import get_alpha, get_PAA,  get_rvo_array, get_beta, cal_exp_tim ,cal_vo_exp_time
# state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
# moving_state_list: [[x, y, z, vx, vy, vz, radius, prb]]其他无人机
# obstacle_state_list: [[x, y, z, radius]]建筑物障碍物
# rvo_vel: [x, y, z, ve_x, ve_y, ve_z, α]速度障碍物存储形式

class rvo_inter(reciprocal_vel_obs):
    def __init__(self, neighbor_region=5, neighbor_num=10, vxmax=1.5, vymax=1.5, acceler=0.5, env_train=True, exp_radius=0.2, ctime_threshold=5, ctime_line_threshold=1, delta_t = 1):
        super(rvo_inter, self).__init__(neighbor_region, vxmax, vymax, acceler)

        self.env_train = env_train
        self.exp_radius = exp_radius
        self.nm = neighbor_num
        self.ctime_threshold = ctime_threshold
        self.ctime_line_threshold = ctime_line_threshold
        self.delta_t = delta_t

    def config_vo_inf(self, robot_state,  odro_state_list, obs_list, action=np.zeros((3,)), **kwargs):
        # mode: vo, rvo, hrvo
        agent_state, ob_list, odro_list= self.preprocess(agent_state, odro_state_list, obs_list)##获取建筑物障碍物与冲突无人机列表

        action = np.squeeze(action)

        vo_list = list(map(lambda x: self.config_vo_circle2(robot_state, x, action, 'rvo', **kwargs), odro_list))#[observation_vo, vo_flag, exp_time, collision_flag, min_dis]

        obs_vo_list = []

        collision_flag = False
        vo_flag = False
        min_exp_time = inf

        for vo_inf in vo_list:
            obs_vo_list.append(vo_inf[0])
            if vo_inf[1] is True:
                # obs_vo_list.append(vo_inf[0])
                vo_flag = True
                if vo_inf[2] < min_exp_time:
                    min_exp_time = vo_inf[2]
            
            if vo_inf[3] is True: collision_flag = True

        obs_vo_list.sort(reverse=True, key=lambda x: (-x[-1], x[-2]))

        if len(obs_vo_list) > self.nm:
            obs_vo_list_nm = obs_vo_list[-self.nm:]
        else:
            obs_vo_list_nm = obs_vo_list

        if self.nm == 0:
            obs_vo_list_nm = []

        return obs_vo_list_nm, vo_flag, min_exp_time, collision_flag
        
    def config_vo_reward(self, robot_state, nei_state_list, obs_cir_list, obs_line_list, action=np.zeros((2,)), **kwargs):

        robot_state, ns_list, oc_list, ol_list = self.preprocess(robot_state, nei_state_list, obs_cir_list, obs_line_list)

        vo_list1 = list(map(lambda x: self.config_vo_circle2(robot_state, x, action, 'rvo', **kwargs), ns_list))
        vo_list2 = list(map(lambda y: self.config_vo_circle2(robot_state, y,action, 'vo', **kwargs), oc_list))
        vo_list3 = list(map(lambda z: self.config_vo_line2(robot_state, z, action, **kwargs), ol_list))

        vo_flag = False
        min_exp_time = inf
        min_dis = inf

        for vo_inf in vo_list1 + vo_list2 + vo_list3:

            if vo_inf[4] < min_dis:
                min_dis = vo_inf[4]

            if vo_inf[1] is True:
                vo_flag = True
                if vo_inf[2] < min_exp_time:
                    min_exp_time = vo_inf[2]

        new_des_list = [des_v[-1] for des_v in vo_list3]
        
        return vo_flag, min_exp_time, min_dis


    def config_vo_observe(self, robot_state, nei_state_list, obs_cir_list, obs_line_list):

        vo_list, _, _, _ = self.config_vo_inf(robot_state, nei_state_list, obs_cir_list, obs_line_list)
        
        return vo_list

    def config_vo_circle2(self, state, odro, action, mode='rvo', **kwargs):
        
        x, y, z, vx, vy, vz, r = state[0:7]#state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
        Pa = [x, y, z]
        Va = [vx, vy, vz]
        mx, my, mz,  mvx, mvy, mvz, mr = odro[0:7]#输入的状态中提取机器人和圆形障碍物的位置、速度和半径。moving_state_list: [[x, y, z, vx, vy, vz, radius, prb]]其他无人机
        Pb = [mx, my, mz]
        Vb = [mvx, mvy, mvz]

        vo_flag = False
        collision_flag = False

        rel_x = mx - x
        rel_y = my - y
        rel_z = mz - z

        dis_mr = sqrt((rel_y)**2 + (rel_x)**2 + sqrt(rel_z)**2)
        
        real_dis_mr = sqrt((rel_y)**2 + (rel_x)**2)
        
        env_train = kwargs.get('env_train', self.env_train)

        if env_train:#根据环境训练标志，检查是否发生碰撞。
            if dis_mr <= r + mr:
                dis_mr = r + mr
                collision_flag = True
        else:
            if dis_mr <= r - self.exp_radius + mr:
                collision_flag = True

            if dis_mr <= r + mr:
                dis_mr = r + mr


        alpha = get_alpha(Pa, Pb, r, mr)
        PAA = get_PAA(Pa, state[7], odro[7], Va, Vb)
        rvo_array = [rel_x, rel_y, rel_z]# rvo_vel: [x, y, z, ve_x, ve_y, ve_z, α]速度障碍物存储形式
        vo = PAA[:3]+rvo_array[:3]+[alpha]#vo: [x, y, z, ve_x, ve_y, ve_z, α]

 
        rel_vx = 2*action[0] - mvx - vx
        rel_vy = 2*action[1] - mvy - vy
        rel_vz = 2*action[2] - mvz - vz


        exp_time = inf

        if self.vo_out_jud_vector(action[0], action[1], action[2], vo):##检测此时无人机速度障碍物
            vo_flag = False
            exp_time = inf
        else:
            exp_time = cal_vo_exp_time(rel_x, rel_y, rel_z, rel_vx,rel_vy,rel_vz, r, mr) 
            if exp_time < self.ctime_threshold:
                vo_flag = True
            else:
                vo_flag = False
                exp_time = inf
            
        input_exp_time = 1 / (exp_time+0.2)
        min_dis = real_dis_mr-mr

        observation_vo = PAA[:3]+rvo_array[:3]+[alpha, min_dis, input_exp_time]
        #速度障碍物的方向和速度。
        #计算期望时间，判断是否存在速度障碍物。
        #构建观测信息，包括速度障碍物的相关参数。
        return [observation_vo, vo_flag, exp_time, collision_flag, min_dis]


    def vo_out_jud_vector(self, agent_state, vx, vy, vz, odro_rvo):##不碰撞返回ture alpha < beta
        vector_out = True
        Panew = [0, 0, 0]
        Panew[0] = agent_state[0] + vx*self.delta_t
        Panew[1] = agent_state[1] + vy*self.delta_t
        Panew[2] = agent_state[2] + vz*self.delta_t
        
        PAA = odro_rvo[0:3]
        rvo_array =  odro_rvo[4:6]
        alpha = odro_rvo[6]
        arr_AA_Anew = []
        for i in range(3):
            arr_AA_Anew.append = Panew[i] - PAA[i]

        beta = get_beta(rvo_array, arr_AA_Anew)
        if alpha > beta:
            vector_out = False
        return vector_out
