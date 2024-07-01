from Drone import Drone
from math import pi, cos, sin ,atan2, pi, sqrt
import numpy as np
import random  
from collections import namedtuple
from ir_sim.util import collision_cir_cir, collision_cir_matrix, collision_cir_seg
# state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
# moving_state_list: [[x, y, z, vx, vy, vz, radius, prb]]其他无人机
# obstacle_state_list: [[x, y, z, radius]]建筑物障碍物
# rvo_vel: [x, y, z, ve_x, ve_y, ve_z, α]速度障碍物存储形式
class env_Drone:##需要输入：无人机数量、航路点、航路点数量、优先级列表
    def __init__(self, Drone_class=Drone, Drone_number=0, step_time=1, components=[],building_list = [], **kwargs):

        self.Drone_class = Drone_class#分类
        self.Drone_number = Drone_number#数量
        self.Drone_list = []
        self.com = components#存储组件列表

        self.interval = kwargs.get('interval', 1)#存储时间间隔，默认为 1
        self.radius = kwargs.get('radius', False)#是否随机半径，默认为 False。

        self.building_list = building_list


        if self.Drone_number > 0:
            assert 'waypoints_list' in kwargs and 'n_points_list' in kwargs and 'priority_list' in kwargs, "Missing required keyword arguments"
            waypoints_list = kwargs['waypoints_list']
            n_pointst_list = kwargs['n_pointst_list']
            priority_list = kwargs['priority_list']
            radius_list = kwargs.get('radius_list', 1)##有默认值
            starting_list, destination_list= self.init_state_distribute(self.init_mode, waypoints_list)
            #使用默认半径列表或从 init_state_distribute 函数中获取初始化状态和目标。

        # 创建无人机对象
        for i in range(self. Drone_number):
            drone = self.Drone_class(id=i, starting = starting_list[i], destination = destination_list[i], waypoints=waypoints_list[i], 
                                      n_points = n_pointst_list[i], init_acc = 1,priority = priority_list[i], step_time=step_time, **kwargs)
            self.Drone_list.append(drone)
            self.drone = drone if i == 0 else None 
        
    def init_state_distribute(self, waypoints_list):#将起点与终点从航路点中提取出来
        starting_list, destination_list = [], []
        for waypoints in waypoints_list:  
            start = waypoints[0]  
            goal = waypoints[-1]  
            starting_list.append(start)  
            destination_list.append(goal)  
        return starting_list, destination_list
    
   
    def distance(self, point1, point2):
        diff = point2[0:3] - point1[0:3]
        return np.linalg.norm(diff)

    # def collision_check(self, components):
 
    #     return False
    
    def collision_check_with_building(self):
        self.collision_flag = False
        for drone in self.Drone_list:
            for building_obj in self.building_list:  
                if drone.state[2] <= building_obj[2]:
                 # 确保无人机在建筑物高度或以下  [xyhr]
                # 计算无人机与建筑物在地面的投影圆的距离  
                    dis = self.distance2D(drone.state, (building_obj[0], building_obj[1]))  
                    if dis <= drone.radius + building_obj[3]:  # 如果距离小于等于两圆半径之和，则碰撞  
                        self.collision_flag = True  
                        print('Drone collided with a building!')  
                        break # 如果发生碰撞，跳出内部循环
            if self.collision_flag:  
                break  # 如果发生碰撞，跳出外部循环  
        return self.collision_flag



    def collision_check_with_drones(self):
        circle = namedtuple('circle', 'x y z r')
        self.collision_flag = False  
        for i, drone in enumerate(self.Drone_list):  
            if drone.collision_flag == True:
                return True
            self_circle = circle(drone.state[0], drone.state[1], drone.state[2], drone.radius)
            for other_drone in self.Drone_list[i+1:]:
                if other_drone is not drone and not other_drone.collision_flag: 
                    other_circle = circle(other_drone.state[0], other_drone.state[1], other_drone.state[2], other_drone.radius) 
                    if self.collision_dro_dro(self_circle, other_circle):  
                        other_drone.collision_flag = True  
                        self.collision_flag = True  
                        print('Drones collided!')  
                        return True 


    def step(self, vel_list=[], **vel_kwargs):

        # vel_kwargs: vel_type = 'diff', 'omni'
        #             stop=True, whether stop when arrive at the goal
        #             noise=False, 
        #             control_std = [0.01, 0.01], noise for omni

        for drone, vel in zip(self.Drone_list, vel_list):
            drone.move_forward(vel, **vel_kwargs)

    def cal_des_list(self):
        vel_list = list(map(lambda x: x.cal_des_vel() , self.Drone_list))
        return vel_list
    
    def arrive_all(self):

        for drone in self.Drone_list:
            if not drone.arrive():
                return False

        return True

    def drones_reset(self, **kwargs):
        for drone in self.Drone_list:
            drone.reset()

    def drone_reset(self, id=0):
        self.Drone_list[id].reset()

    def total_states(self):
        drone_state_list = list(map(lambda r: np.squeeze( r.dronestate()), self.Drone_list))# state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
        return drone_state_list
    

    @staticmethod  
    def collision_dro_dro(circle1, circle2):  
        dis = Drone.distance(circle1, circle2)   
        if 0 < dis <= circle1.r + circle2.r:  
            return True  
        return False
    
    @staticmethod
    def distance(point1, point2):
        distance = sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(point1[2]-point2[2])**2)
        return distance
    
    @staticmethod
    def distance2D(point1, point2):
        distance = sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
        return distance
    # # states
    # def total_states(self, env_train=True):
        
    #     robot_state_list = list(map(lambda r: np.squeeze( r.omni_state(env_train)), self.robot_list))
    #     nei_state_list = list(map(lambda r: np.squeeze( r.omni_obs_state(env_train)), self.robot_list))
    #     obs_circular_list = list(map(lambda o: np.squeeze( o.omni_obs_state(env_train) ), self.obs_cir_list))
    #     obs_line_list = self.obs_line_list
        
    #     return [robot_state_list, nei_state_list, obs_circular_list, obs_line_list]
        
    # def render(self, time=0.1, save=False, path=None, i = 0, **kwargs):
        
    #     self.world_plot.draw_robot_diff_list(**kwargs)
    #     self.world_plot.draw_obs_cir_list()
    #     self.world_plot.pause(time)

    #     if save == True:
    #         self.world_plot.save_gif_figure(path, i)

    #     self.world_plot.com_cla()

    
    # def seg_dis(self, segment, point):
        
    #     point = np.squeeze(point[0:2])
    #     sp = np.array(segment[0:2])
    #     ep = np.array(segment[2:4])

    #     l2 = (ep - sp) @ (ep - sp)

    #     if (l2 == 0.0):
    #         return np.linalg.norm(point - sp)

    #     t = max(0, min(1, ((point-sp) @ (ep-sp)) / l2 ))

    #     projection = sp + t * (ep-sp)

    #     distance = np.linalg.norm(point - projection) 

    #     return distance