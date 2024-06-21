import numpy as np
from math import sin, cos, atan2, pi, sqrt
from line_sight_partial_3D import line_sight_partial_3D
from collections import namedtuple
# state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
# moving_state_list: [[x, y, z, vx, vy, vz, radius, prb]]其他无人机
# obstacle_state_list: [[x, y, z, radius]]建筑物障碍物
# rvo_vel: [x, y, z, ve_x, ve_y, ve_z, α]速度障碍物存储形式
import numpy as np  
  
class Drone:  
    def __init__(self, id, starting, destination, waypoints, n_points, init_acc,components, dt=1, init_state=np.zeros((3, 1)),    
                 vel=np.zeros((3, 1)), vel_max=2*np.ones((3, 1)), goal_threshold=0.1, radius=5, **kwargs):  
  
        self.id = int(id)  # 无人机编号  
        self.state = np.array(init_state, ndmin=2) if not isinstance(init_state, np.ndarray) else init_state  # 初始位置  
        self.previous_state = self.state.copy()  
  
        self.vel = np.array(vel, ndmin=2) if not isinstance(vel, np.ndarray) else vel  # 速度  
        self.vel_max = np.array(vel_max, ndmin=2) if not isinstance(vel_max, np.ndarray) else vel_max  # 速度上限  
  
        # 转换starting和destination为NumPy数组  
        self.starting = np.array(starting, ndmin=2) if isinstance(starting, list) else starting  
        self.destination = np.array(destination, ndmin=2) if isinstance(destination, list) else destination  
  
        # waypoints作为列表传入，无需转换为NumPy数组  
        self.waypoints = waypoints  
        self.n_points = n_points  # 航路点个数  
  
        # 当前目标点，初始化为destination或者waypoints的第二个点 （waypoints）
        if len(waypoints) > 0:  
            self.current_des = np.array(waypoints[1], ndmin=2)  
        else:  
            self.current_des = destination

        self.i = 1  
  
        self.acc = init_acc  # 加速度  
        self.dt = dt  # 时间步长  
        self.radius = radius  # 半径  
  
        self.arrive_flag = False  # 到达标志  
        self.collision_flag = False  # 碰撞标志  
        self.goal_threshold = goal_threshold  # 到达目标的阈值 

        self.components = components 
        self.radius_collision = round(radius + kwargs.get('radius_exp', 1), 2)  # 碰撞检测半径
        
        # 添加noise参数，如果kwargs中没有提供，则默认为False  
        self.__noise = kwargs.get('noise', False)  
  
        # 添加control_std参数，如果kwargs中没有提供，则使用默认值  
        self.__control_std = kwargs.get('control_std', [0.01, 0.01, 0.01])  

  

    def update_info(self, state, vel):
    # 更新状态
        self.state = state
        self.vel = vel

    def move_forward(self, vel, stop=True, **kwargs): 
    
        if isinstance(vel, list):
            vel = np.array(vel, ndmin=2).T

        if vel.shape == (3,):
            vel = vel[:,np.newaxis]

        assert vel.shape == (3, 1)

        vel = np.clip(vel, -self.vel_max, self.vel_max)#限制范围
        
        if stop:
            if self.arrive_flag or self.collision_flag:
                vel = np.zeros((3, 1))


        self.previous_state = self.state
        self.move(vel, self.__noise, self.__control_std)

        self.arrive(self.state, self.current_des)


    def current_des_new(self, E3d, map_size):  
        if self.arrive(self.state, self.current_des):  
            if self.i < self.n_points - 1:  # 减1是因为当前已经到达了第i个点  
                next_waypoint = self.waypoints[self.i + 1]  
                if line_sight_partial_3D(E3d, (self.state[0], next_waypoint[0]),  (self.state[1], next_waypoint[1]),  (self.state[2], next_waypoint[2]),  map_size) == 1:  
                    self.i += 1  
                    self.current_des = np.array(next_waypoint, ndmin=2)  
            else:  
                self.arrive_flag = True

    def move(self, vel, noise=False, std=None):  
        if std is None:  
            std = self.__control_std
        next_state = self.motion(self.state, vel, self.dt, noise, std)  
        self.state = next_state  
        self.vel = vel 


    def motion(current_state, vel, sampletime, noise = False, control_std = None):
        if control_std is None:  
            control_std = [0.01, 0.01, 0.01]
    # vel: np.array([[vel x], [vel y],[vel z]])
    # current_state: np.array([[x], [y], [z]])

        if noise:  
            vel_noise = vel + np.random.normal(np.zeros((3, 1)), scale=np.array(control_std).reshape((3, 1)))  
        else:  
            vel_noise = vel  
  
        next_state = current_state + vel_noise * sampletime  
        return next_state

    def arrive(self, current_position, current_des):

        position = current_position[0:3]
        dist = np.linalg.norm(position - current_des[0:3]) 

        if dist < self.goal_threshold:
            self.arrive_flag = True
            return True
        else:
            self.arrive_flag = False
            return False 
    
    def move_to_goal(self):
        vel = self.cal_des_vel()
        self.move_forward(vel) 

    
    def cal_des_vel(self):  
        dis, angles = self.relative(self.state[0:3], self.destination)  
          
        if dis > self.goal_threshold:  
            # 假设 self.vel_max 是一个长度为3的列表或numpy数组，代表x, y, z三个方向的最大速度  
            # 将角度转换为单位方向向量  
            dir_vector = self.angles_to_direction(angles)  
              
            # 缩放单位方向向量到最大速度  
            vel_scaled = np.multiply(self.vel_max, dir_vector)  
            vel = vel_scaled.reshape((3, 1))  # 确保是(3, 1)形状的  
        else:    
            vel = np.zeros((3, 1)) 
  
        return vel
    
############################################## circle = namedtuple('circle', 'x y z r')这里闹不清楚
    def collision_check(self, components):
        circle = namedtuple('circle', 'x y z r')
        building_obj = namedtuple('building', 'x y h r')
        
        self_circle = circle(self.state[0], self.state[1], self.state[2], self.radius)
        
        if self.collision_flag == True:
            return True

        #检查无人机间冲突
        for other_drone in components['Drones'].drone_list:
            if other_drone is not self and not other_drone.collision_flag: 
                other_circle = circle(other_drone.state[0], other_drone.state[1], other_drone.state[2], other_drone.radius) 
                if self.collision_dro_dro(self_circle, other_circle):  
                    other_drone.collision_flag = True  
                    self.collision_flag = True  
                    print('Drones collided!')  
                    return True 
       # 检查无人机与圆柱建筑物冲突  
        for building_obj in components['building'].building_list:  
            if self.state[2] <= building_obj.h:  # 确保无人机在建筑物高度或以下  
                # 计算无人机与建筑物在地面的投影圆的距离  
                dis = self.distance2D(self_circle, (building_obj.x, building_obj.y))  
                if dis <= self_circle.r + building_obj.r:  # 如果距离小于等于两圆半径之和，则碰撞  
                    self.collision_flag = True  
                    print('Drone collided with a building!')  
                    return True  
  
        # 如果没有碰撞，返回False  
        return False 

    def set_state(self):
        v_des = self.cal_des_vel()
        rc_array = self.radius_collision * np.ones((1,1))
            
        return np.concatenate((self.state[0:3], self.vel, rc_array, v_des), axis = 0)

    def obs_state(self):
        rc_array = self.radius * np.ones((1,1))
        return np.concatenate((self.state[0:3], self.vel, rc_array), axis = 0) 


    def reset(self, random_bear=False):

        self.state[:] = self.init_state[:]
        self.previous_state[:] = self.init_state[:]
        self.vel = np.zeros((3, 1))
        self.arrive_flag = False
        self.collision_flag = False

        if random_bear:
            self.state[3, 0] = np.random.uniform(low = -pi, high = pi)

    # 检查两个无人机是否碰撞  
    @staticmethod  
    def collision_dro_dro(circle1, circle2):  
        dis = Drone.distance(circle1, circle2)   
        if 0 < dis <= circle1.r + circle2.r:  
            return True  
        return False
    

    # 检查无人机是否与建筑物的圆形投影碰撞  
    @staticmethod  
    def collision_dro_building(drone_circle, building_projection):  
        dis = Drone.distance2D(drone_circle, building_projection)  # 假设这个函数可以计算无人机与建筑物水平投影之间的距离  
        if 0 < dis <= drone_circle.r + building_projection.r:  
            return True  
        return False


    @staticmethod  
    def angles_to_direction(angles):  
        azimuth, elevation = angles  
        # 将方位角和俯仰角转换为单位方向向量  
        dir_vector = np.array([  
            np.cos(azimuth) * np.cos(elevation),  
            np.sin(azimuth) * np.cos(elevation),  
            np.sin(elevation)  
        ])  
        return dir_vector
    
    @staticmethod
    def distance(point1, point2):
        distance = sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(point1[2]-point2[2])**2)
        return distance
    
    @staticmethod
    def distance2D(point1, point2):
        distance = sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
        return distance
                 
    @staticmethod  
    def relative(state1, state2):  
        dif = np.array(state2[0:3]) - np.array(state1[0:3])  
  
        dis = np.linalg.norm(dif)  # 计算向量长度（范数）  
          
        # 计算与x轴的夹角（方位角）和与z轴的夹角（俯仰角）  
        # 这里使用np.arctan2来避免除以0的情况，并自动处理四个象限  
        azimuth = np.arctan2(dif[1], dif[0])  # 方位角（绕z轴旋转）  
        elevation = np.arctan2(dif[2], np.linalg.norm(dif[0:2]))  # 俯仰角（绕y轴旋转）
        
        if dis != 0:  # 防止除以零  
            elevation = np.arctan2(dif[2], np.linalg.norm(dif[0:2]))  # 俯仰角（绕y轴旋转）  
        else:  
            elevation = 0  # 如果距离为零，俯仰角可以定义为零（或者任意值，因为它不影响方向）  
    
        return dis, (azimuth, elevation) 
    

    @staticmethod
    def to_pi(radian):

        if radian > pi:
            radian = radian - 2 * pi
        elif radian < -pi:
            radian = radian + 2 * pi
        
        return radian
    


    """ def cal_des_vel_omni(self):  
        dis, radian = self.relative(self.state[0:3], self.goal)  
          
        if dis > self.goal_threshold:  
            # 假设 self.vel_max 是一个长度为3的列表或numpy数组，代表x, y, z三个方向的最大速度  
            vx = self.vel_max[0] * np.cos(radian[0]) * np.cos(radian[1])  
            vy = self.vel_max[1] * np.sin(radian[0]) * np.cos(radian[1])  
            vz = self.vel_max[2] * np.sin(radian[1])  
        else:  
            vx = 0  
            vy = 0  
            vz = 0  
  
        return np.array([vx, vy, vz])  
  
    @staticmethod  
    def relative(state1, state2):  
        dif = np.array(state2[0:3]) - np.array(state1[0:3])  
  
        dis = np.linalg.norm(dif)  # 计算向量长度（范数）  
          
        # 计算与x轴的夹角（方位角）和与z轴的夹角（俯仰角）  
        # 这里使用np.arctan2来避免除以0的情况，并自动处理四个象限  
        azimuth = np.arctan2(dif[1], dif[0])  # 方位角（绕z轴旋转）  
        elevation = np.arctan2(np.linalg.norm(dif[0:2]), dif[2])  # 俯仰角（绕y轴旋转）  
          
        # 将方位角和俯仰角转换为单位向量的方向角（球坐标到笛卡尔坐标）  
        # 注意这里我们假设无人机是在世界坐标系中移动，而不是在其自身的机体系中  
        dir_vector = np.array([  
            np.cos(azimuth) * np.cos(elevation),  
            np.sin(azimuth) * np.cos(elevation),  
            np.sin(elevation)  
        ])  
          
        # 如果只需要角度而不是方向向量，可以返回 azimuth 和 elevation  
        # 但为了与原始代码保持一致，我们返回方向向量的角度（这里简化为只返回 azimuth）  
        # 在实际应用中，可能需要更复杂的处理来考虑三维空间中的运动  
        # 例如，可以使用方向向量的球面坐标 (azimuth, elevation) 或者直接返回 dir_vector  
        return dis, azimuth  # 或者返回 dis, (azimuth, elevation) 或者 dis, dir_vector  
  
# 注意：在实际应用中，可能还需要考虑无人机的当前姿态（如翻滚、偏航和俯仰角）  
# 来计算真正的期望速度向量。这里的代码仅提供了基本的计算框架 """