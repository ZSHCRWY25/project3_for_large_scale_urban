import yaml
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from PIL import Image
from pynput import keyboard
import xlrd
from path_planning_main import path_planning_main as pathplan
from grid_3D_safe_zone import grid_3D_safe_zone

class env_base:

    def __init__(self, map_height = 500, map_width = 500, map_high =30 , dron_num = 8, **kwargs):
        self.map_height = map_height
        self.map_width = map_width
        self.map_high = map_high
        self.map_size = [map_height, map_width, map_high]
        self.dron_num = dron_num
        self.starting = []
        self.destination = []
        self.components = dict()


    def init_map(self):
        E, E_safe, E3d, E3d_safe ,obs_list = grid_3D_safe_zone(self.map_size, 1, 25, self.dron_num, self.starting, self.destination, 2)
    #E是二维障碍物地图
    #E_safe带保护区用来画图
    #E3d三维障碍物，用于学习环境
    #E3d_safe带保护区，用于路径规划
    #obs = [[x,y,h,r]]
        return E, E_safe, E3d, E3d_safe ,obs_list

    def init_road(self, E3d_safe):
        for i in range(self.dron_num):
            path, n_pionts = pathplan(self.map_size, self.starting[i], self.destination[i], E3d_safe)
        
        return path, n_pionts


    def init_environment(self, robot_class=mobile_robot, obs_cir_class=obs_circle, **kwargs):

 


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


    



        
        
            
    
        

        

