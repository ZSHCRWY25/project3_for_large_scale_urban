'''
:@Author: 刘玉璞
:@Date: 2024/6/10 16:43:44
:@LastEditors: 刘玉璞
:@LastEditTime: 2024/7/2 16:43:44
:Description: 
'''
import os
import yaml
import numpy as np
import sys
import math
from envs.env.env_plot import env_plot
import xlrd
from envs.env.env_drones import env_Drone
from envs.env.Drone import Drone
from envs.world.path_planning_main import path_planning_main as pathplan##这里还要改
from envs.world.grid_3D_safe_zone import grid_3D_safe_zone

class env_base:

    def __init__(self, map_height = 50, map_width = 50, map_high =10 , **kwargs):
        self.map_height = map_height
        self.map_width = map_width
        self.map_high = map_high
        self.map_size = [map_height, map_width, map_high]
        self.components = dict()
        self.building_list = []
        self.waypoints_list = []
        self.n_points_list = []
        self.priority_list = []
        self.starting = []
        self.destination = []
        self.robots_args = []
        self.plot = True
        self.init_environment(drone_class=Drone)

    def read_start_des(self):
        current_dir = os.getcwd()  
        # 构建.drone_paths.yaml文件的完整路径  
        file_path = os.path.join(current_dir, 'gym_env\\envs\\world\\drone_paths.yaml')  

        # 检查文件是否存在  
        if not os.path.exists(file_path):  
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        try:  
            # 打开并读取YAML文件  
            with open(file_path, 'r') as file:

                data = yaml.safe_load(file)  
                # 从数据中提取起点和终点的列表  
                self.starting  = data['start_points']  
                self.destination = data['end_points']
                self.dron_num = len(self.starting)

        except yaml.YAMLError as e:  
            # 处理YAML格式错误  
            raise ValueError(f"Error parsing YAML file {file_path}: {e}")  
      
        except Exception as e:  
            # 处理其他未预期的异常  
            raise RuntimeError(f"An unexpected error occurred while reading the YAML file: {e}")  
  


    def init_map_road(self):
        E, E_safe, E3d, E3d_safe ,obs_list = grid_3D_safe_zone(self.map_size, 1, 10, self.dron_num, self.starting, self.destination, 1)
    #E是二维障碍物地图
    #E_safe带保护区用来画图
    #E3d三维障碍物，用于学习环境
    #E3d_safe带保护区，用于路径规划
    #obs = [[x,y,h,r]]
        self.E3d = E3d
        self.building_list = obs_list
        for i in range(self.dron_num):
            path, n_pionts = pathplan(self.map_size, self.starting[i], self.destination[i], E3d_safe)
            self.waypoints_list.append(path)
            self.n_points_list.append(n_pionts)

        # waypoints_list = kwargs['waypoints_list']
        # n_pointst_list = kwargs['n_pointst_list']
        # priority_list = kwargs['priority_list']


    def init_environment(self, drone_class=Drone):

        self.read_start_des()
        self.init_map_road()

        self.components['drones'] = env_Drone(waypoints_list = self.waypoints_list,n_points_list = self.n_points_list,priority_list = self.priority_list,
                                              building_list = self.building_list, drone_class=drone_class, Drone_number=self.dron_num, step_time=1, 
                                              )
        self.drone_list = self.components['drones'].Drone_list
        #assert 'waypoints_list' in kwargs and 'n_points_list' in kwargs and 'priority_list' in kwargs, "Missing required keyword arguments"

        self.time = 0
        if self.dron_num> 0:
            self.Drone = self.components['drones'].Drone_list[0]
        

        if self.plot:
            self.world_plot = env_plot(self.map_size, self.building_list, self.components)
        
        self.time = 0
        



    def collision_check(self):##检查碰撞
        collision = False
        for drone in self.components['drones'].Drone_list: 
            if drone.collision_check_with_dro(self.components):
                collision = True
            if drone.collision_check_with_building(self.building_list):
                collision = True
        return collision
    
    def arrive_check(self):##检查到没到
        arrive = False

        for drone in self.components['drones'].Drone_list: 
            if not drone.arrive_flag:
                arrive = True

        return arrive

    def drone_step(self, vel_list, drone_id = None, **kwargs):

        if drone_id == None:
            if not isinstance(vel_list, list):
                self.Drone.move_forward(vel_list, self.E3d,self.map_size, **kwargs)
            else:
                for i, drone in enumerate(self.components['drones'].Drone_list):
                   drone.move_forward(vel_list[i],self.E3d,self.map_size, **kwargs)
        else:
            self.components['drones'].Drone_list[drone_id-1].move_forward(vel_list, **kwargs)

    def see_des(self, drone_id = None):
        if drone_id == None:
            for drone in enumerate(self.components['drones'].Drone_list):
                drone.if_see_des(self.E3d, self.map_size)
            else:
                self.components['drones'].Drone_list[drone_id-1].if_see_des(self.E3d, self.map_size)



    def render(self, time=0.05, **kwargs):

        if self.plot:
            self.world_plot.clear_plot_elements()
            self.world_plot.draw_drones(**kwargs)
            self.world_plot.pause(time)
            
        self.time = self.time + time
        
    def save_fig(self, path, i):
        self.world_plot.save_gif_figure(path, i)
    
    def save_ani(self, image_path, ani_path, ani_name='animated', **kwargs):
        self.world_plot.create_animate(image_path, ani_path, ani_name=ani_name, **kwargs)

    def show(self, **kwargs):
        self.world_plot.draw_drones(**kwargs)
        self.world_plot.show()
    
    def show_ani(self):
        self.world_plot.show_ani()
    

    



        
        
            
    
        

        

