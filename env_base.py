import yaml
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from PIL import Image
from pynput import keyboard
import xlrd
from path_planning_main import path_plan
from grid_3D_safe_zone import grid_3D_safe_zone

class env_base:

    def __init__(self, map_height = 500, map_width = 500, map_high =30 , dron_num = 8, **kwargs):
        self.map_height = map_height
        self.map_width = map_width
        self.map_high = map_high
        self.dron_num = dron_num
        self.starting = []
        self.destination = []


    def init_map(self):
        map_size = [self.map_height, self.map_width,self.map_high]
        [E, E_safe, E3d, E3d_safe ,obs_list] = grid_3D_safe_zone(map_size, 1, 25, self.dron_num, self.starting, self.destination, 2)


    def init_environment(self, robot_class=mobile_robot, obs_cir_class=obs_circle, **kwargs):

        # world
        px = int(self.__width / self.xy_reso)#计算了世界地图的像素大小(px和py)
        py = int(self.__height / self.xy_reso)

        if self.world_map != None:
            
            world_map_path = sys.path[0] + '/' + self.world_map
            img = Image.open(world_map_path).convert('L')
            # img = Image.open(world_map_path)
            img = img.resize( (px, py), Image.NEAREST)#读取图像并将其转换为灰度图像（'L’表示灰度
            # img = img.resize( (px, py), Image.ANTIALIAS)
            # img.thumbnail( (px, py))

            map_matrix = np.array(img)
            map_matrix = 255 - map_matrix
            map_matrix[map_matrix>255/2] = 255#二值化处理，将像素值大于255/2的设为255，小于255/2的设为0
            map_matrix[map_matrix<255/2] = 0
            # map_matrix[map_matrix>0] = 255
            # map_matrix[map_matrix==0] = 0

            self.map_matrix = np.fliplr(map_matrix.T)
        else:
            self.map_matrix = None
        
        
        ##以下是获取环境中障碍物信息

        self.components['map_matrix'] = self.map_matrix
        self.components['xy_reso'] = self.xy_reso
        self.components['offset'] = np.array([self.offset_x, self.offset_y])

        self.components['obs_lines'] = env_obs_line(**{**self.obs_lines_args, **kwargs})
        self.obs_line_states=self.components['obs_lines'].obs_line_states

        self.components['obs_circles'] = env_obs_cir(obs_cir_class=obs_cir_class, obs_cir_num=self.obs_cir_number, step_time=self.step_time, components=self.components, **{**self.obs_cirs_args, **kwargs})
        self.obs_cir_list = self.components['obs_circles'].obs_cir_list

        self.components['obs_polygons'] = env_obs_poly(obs_poly_class=obs_polygon_class, vertex_list=self.vertexes_list, obs_poly_num=self.obs_poly_num, **{**self.obs_polygons_args, **kwargs})
        self.obs_poly_list = self.components['obs_polygons'].obs_poly_list        

        self.components['robots'] = env_robot(robot_class=robot_class, step_time=self.step_time, components=self.components, **{**self.robots_args, **kwargs})
        self.robot_list = self.components['robots'].robot_list

        self.components['cars'] = env_car(car_class=car_class, car_num=self.car_number, step_time=self.step_time, **{**self.cars_args, **kwargs})
        self.car_list = self.components['cars'].car_list

        if self.plot:
            self.world_plot = env_plot(self.__width, self.__height, self.components, offset_x=self.offset_x, offset_y=self.offset_y, **kwargs)
        
        self.time = 0

        if self.robot_number > 0:
            self.robot = self.components['robots'].robot_list[0]
        
        if self.car_number > 0:
            self.car = self.components['cars'].car_list[0]
    
    def collision_check(self):##检查与车碰撞
        collision = False
        for robot in self.components['robots'].robot_list: 
            if robot.collision_check(self.components):
                collision = True

        for car in self.components['cars'].car_list: 
            if car.collision_check(self.components):
                collision = True

        return collision
    
    def arrive_check(self):##检查到没到
        arrive=True

        for robot in self.components['robots'].robot_list: 
            if not robot.arrive_flag:
                arrive = False

        for car in self.components['cars'].car_list: 
            if not car.arrive_flag:
                arrive = False

        return arrive

    def robot_step(self, vel_list, robot_id = None, **kwargs):

        if robot_id == None:

            if not isinstance(vel_list, list):
                self.robot.move_forward(vel_list, **kwargs)
            else:
                for i, robot in enumerate(self.components['robots'].robot_list):
                    robot.move_forward(vel_list[i], **kwargs)
        else:
            self.components['robots'].robot_list[robot_id-1].move_forward(vel_list, **kwargs)

        for robot in self.components['robots'].robot_list:
            robot.cal_lidar_range(self.components)

    def car_step(self, vel_list, car_id=None, **kwargs):

        if car_id == None:
            if not isinstance(vel_list, list):
                self.car.move_forward(vel_list, **kwargs)
            else:
                for i, car in enumerate(self.components['cars'].car_list):
                    car.move_forward(vel_list[i], **kwargs)
        else:
            self.components['cars'].car_list[car_id-1].move_forward(vel_list, **kwargs)
        
        for car in self.components['cars'].car_list:
            car.cal_lidar_range(self.components)

    def obs_cirs_step(self, vel_list=[], obs_id=None, **kwargs):
        
        if self.obs_step_mode == 'default':
            if obs_id == None:
                for i, obs_cir in enumerate(self.components['obs_circles'].obs_cir_list):
                    obs_cir.move_forward(vel_list[i], **kwargs)
            else:
                self.components['obs_circles'].obs_cir_list[obs_id-1].move_forward(vel_list, **kwargs)

        elif self.obs_step_mode == 'wander':
            # rvo
            self.components['obs_circles'].step_wander(**kwargs)

    def render(self, time=0.05, **kwargs):

        if self.plot:
            self.world_plot.com_cla()
            self.world_plot.draw_dyna_components(**kwargs)
            self.world_plot.pause(time)
            
        self.time = self.time + time

    # def reset(self, ):


    def on_press(self, key):

        try:

            if key.char.isdigit() and self.alt_flag:

                if int(key.char) > self.robot_number:
                    print('out of number of robots')
                else:
                    self.key_id = int(key.char)

            if key.char == 'w':
                self.key_lv = self.key_lv_max
            if key.char == 's':
                self.key_lv = - self.key_lv_max
            if key.char == 'a':
                self.key_ang = self.key_ang_max
            if key.char == 'd':
                self.key_ang = -self.key_ang_max
            
            self.key_vel = np.array([self.key_lv, self.key_ang])

        except AttributeError:
            
            if key == keyboard.Key.alt:
                self.alt_flag = 1
    
    def on_release(self, key):
        
        try:
            if key.char == 'w':
                self.key_lv = 0
            if key.char == 's':
                self.key_lv = 0
            if key.char == 'a':
                self.key_ang = 0
            if key.char == 'd':
                self.key_ang = 0
            if key.char == 'q':
                self.key_lv_max = self.key_lv_max - 0.2
                print('current lv ', self.key_lv_max)
            if key.char == 'e':
                self.key_lv_max = self.key_lv_max + 0.2
                print('current lv ', self.key_lv_max)
            
            if key.char == 'z':
                self.key_ang_max = self.key_ang_max - 0.2
                print('current ang ', self.key_ang_max)
            if key.char == 'c':
                self.key_ang_max = self.key_ang_max + 0.2
                print('current ang ', self.key_ang_max)
            
            self.key_vel = np.array([self.key_lv, self.key_ang])

        except AttributeError:
            if key == keyboard.Key.alt:
                self.alt_flag = 0

        
    def save_fig(self, path, i):
        self.world_plot.save_gif_figure(path, i)
    
    def save_ani(self, image_path, ani_path, ani_name='animated', **kwargs):
        self.world_plot.create_animate(image_path, ani_path, ani_name=ani_name, **kwargs)

    def show(self, **kwargs):
        self.world_plot.draw_dyna_components(**kwargs)
        self.world_plot.show()
    
    def show_ani(self):
        self.world_plot.show_ani()
    
    def read_coordinates_from_excel(file_path):
        """
        从 Excel 文件中读取无人机的起点坐标和终点坐标。

    Args:
        file_path (str): Excel 文件路径。

    Returns:
        tuple: 包含两个列表的元组，第一个列表是起点坐标，第二个列表是终点坐标。
    """
        f1origin = []
        f2origin = []

        try:
            data = xlrd.open_workbook(file_path)
            nums = len(data.sheets())

            for i in range(nums):
                sheet = data.sheets()[i]
                nrows = sheet.nrows

                for row in range(nrows):
                    origin = sheet.row_values(row)[:3]
                    destination = sheet.row_values(row)[3:]

                    f1origin.append(origin)
                    f2origin.append(destination)

            return f1origin, f2origin
        except Exception as e:
            print(f"Error reading coordinates from Excel: {e}")
            return None, None


    



        
        
            
    
        

        

