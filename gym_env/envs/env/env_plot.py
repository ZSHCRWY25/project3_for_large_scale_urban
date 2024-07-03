'''
:@Author: 刘玉璞
:@Date: 2024/6/30 16:43:54
:@LastEditors: 刘玉璞
:@LastEditTime: 2024/7/2 16:43:54
:Description: 
'''
import numpy as np
import matplotlib as mpl
import matplotlib.animation as animation
import imageio
import platform
import shutil
from matplotlib import image
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
from mpl_toolkits.mplot3d import proj3d  
from math import cos, sin, pi
from pathlib import Path
import inspect
import matplotlib.transforms as mtransforms
from matplotlib.animation import FuncAnimation  
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  
from PIL import Image

class env_plot:
    def __init__(self, map_size, building_list, components, lenth= 10, width=10, height=10,
                 full=False, keep_path=False, map_matrix=None, offset_x = 0, offset_y=0):

        self.fig = plt.figure()  
        self.ax = self.fig.add_subplot(111, projection='3d')  
        
        self.width = width
        self.lenth = lenth
        self.height = height


        self.offset_x = offset_x
        self.offset_y = offset_y

        self.color_list = ['g', 'b', 'r', 'c', 'm', 'y', 'k', 'w']
        self.components = components

        self.keep_path=keep_path
        self.map_matrix = map_matrix#地图

        self.building_list = building_list
        self.map_size = map_size
        self.drone_plot_list = []
        self.text_list = [] 
        self.vel_line_list = []
        self.trajectory_line_list = []


        self.init_plot()

        if full:#full为True，则尝试将图形窗口设置为全屏模式（根据操作系统）。
            mode = platform.system()
            if mode == 'Linux':
                plt.get_current_fig_manager().full_screen_toggle()
            elif mode == 'Windows':
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()

    def init_plot(self):##没改
        self.ax.set_aspect('auto')#确保x和y轴的比例相
        self.ax.set_xlim(self.offset_x, self.offset_x + self.width)#设置x和y轴的限制，使用offset_x和offset_y进行偏移
        self.ax.set_ylim(self.offset_y, self.offset_y + self.height)
        self.ax.set_xlabel("x [m]")#设置轴的标签
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m]")

        current_file_frame = inspect.getfile(inspect.currentframe())
        drone_image_path = Path(current_file_frame).parent / 'drone0.png'
        self.drone_img = Image.open(drone_image_path)  
        self.drone_img_box = OffsetImage(self.drone_img, zoom=0.05)

        self.plot_buildings_on_map(self.map_size,self.building_list)#绘制环境的静态组件
        drones = self.components['drones'].Drone_list
        for drone in drones:
            self.draw_waypoints(drone)    
        return self.ax.patches + self.ax.texts + self.ax.artists


    def plot_buildings_on_map(self, map_size, building_list):  
        # 地图大小可能包含x、y和z轴的范围，但在这里我们只关心x和y的范围  
        x_range, y_range, z_range = map_size
        
        fig = plt.figure(figsize=(8, 6))  
        ax = fig.add_subplot(111, projection='3d')  
      
        # 设置地图的x和y轴范围  
        ax.set_xlim(0, x_range)  
        ax.set_ylim(0, y_range)  
        ax.set_zlim(0, z_range + 5)  # z轴范围基于建筑物最高度+一点额外空间  
      
        # 绘制每个建筑物（圆柱体）  
        for b in building_list:  
            x, y, h, r = b  
          
            # 生成极坐标和高度  
            u = np.linspace(0, 2 * np.pi, 50)  
            h_vals = np.linspace(0, h, 20)  # 足够的高度切片以形成平滑的圆柱体  
          
         # 使用meshgrid生成二维网格上的X, Y, Z  
            U, H = np.meshgrid(u, h_vals)  
            X = x + r * np.sin(U)  # 考虑建筑物的x坐标  
            Y = y + r * np.cos(U)  # 考虑建筑物的y坐标  
            Z = H  
          
            # 绘制曲面，并设置颜色为蓝色  
            ax.plot_surface(X, Y, Z, linewidth=0, facecolor='b', shade=True, alpha=0.6)
            plt.gca().set_prop_cycle(None)   
      
    def draw_waypoints(self, drone):
        if len(drone.waypoints)>2:
            middle_waypoints = drone.waypoints[1:-1]  
 
            x = [wp[0] for wp in middle_waypoints]  
            y = [wp[1] for wp in middle_waypoints]  
            z = [wp[2] for wp in middle_waypoints] 
            ax = plt.figure().add_subplot(111,projection='3d')

            ax.scatter(x, y, z,c='r',marker='o')
            for i in range(len(x)):
                ax.text(x[i],y[i],z[i],str(i), zdir='y')
        else:
            return 0


    def draw_drones(self):#改完
        drones = self.components.get('drones', None)
        for drone in drones.Drone_list:
            self.draw_drone(drone)



    def draw_drone(self, drone, drone_color = 'g', destination_color='r', show_vel=True, show_goal=True, show_text=True, show_traj=False, traj_color='-g', **kwargs):
        
        x, y, z = drone.state[:3]  
        # 绘制无人机位置（使用散点图或其他方法）  
        a = self.ax.scatter(x, y, z, c=drone_color, marker='^', label=f'Drone {drone.id}')
        self.drone_plot_list.append(a)
  

        if show_text:  
            b = self.ax.text(x - 0.5, y, z, 'D' + str(drone.id), fontsize=10, color='k',  zorder=5)  # 同样确保文本可见  )
            self.text_list.append(b)  
  
        if show_traj:  
            x_list = [drone.previous_state[0], drone.state[0]]  
            y_list = [drone.previous_state[1], drone.state[1]]  
            z_list = [drone.previous_state[2], drone.state[2]]  
            c = self.ax.plot(x_list, y_list, z_list, color = traj_color)[0]
            self.trajectory_line_list.append(c) 
  
        if show_vel:  
            vel_x, vel_y, vel_z = drone.vel   
            d = self.draw_vector( x, y, z, vel_x, vel_y, vel_z, color='r') 
            self.vel_line_list.append(d)
          



    def draw_vector(self, x, y, z, vel_x, vel_y, vel_z, color='r'):  
         # 计算速度向量的模（长度）  (可以归一化箭头长度)
        vel_norm = np.linalg.norm([vel_x, vel_y, vel_z])  
      
        # 如果速度为0，则不绘制箭头  
        if vel_norm == 0:  
            return None  
      
        scale_factor = 0.2  
        dx, dy, dz = vel_x * scale_factor, vel_y * scale_factor, vel_z * scale_factor
        # 绘制箭头
        a = self.ax.quiver(x,y,z,dx,dy,dz)
        return a

    def draw_trajectory(self, traj, color='g', label='trajectory', show_direction=False, refresh=False):  ############改到这里
        # traj 应该是一个形状为 (num_points, 3) 的 NumPy 数组  
        path_x_list = traj[:, 0]  
        path_y_list = traj[:, 1]  
        path_z_list = traj[:, 2]  
  
        # 绘制轨迹线  
        line = self.ax.plot(path_x_list, path_y_list, path_z_list, color=color, label=label)  
  
        if show_direction:  
            # 假设traj的最后一个元素包含了方向（单位向量），我们需要先计算方向向量的长度  
            if traj.shape[1] > 3:  
                # 假设最后三个元素是dx, dy, dz（方向向量的分量）  
                dx_list = traj[:, -3] / np.linalg.norm(traj[:, -3:], axis=1, keepdims=True)  
                dy_list = traj[:, -2] / np.linalg.norm(traj[:, -3:], axis=1, keepdims=True)  
                dz_list = traj[:, -1] / np.linalg.norm(traj[:, -3:], axis=1, keepdims=True)  
  
                # 在每个点上绘制方向箭头  
                for i in range(len(traj)):  
                    x, y, z = traj[i, :3]  # 提取点的坐标  
                    dx, dy, dz = dx_list[i], dy_list[i], dz_list[i]  # 提取方向向量的分量  
                    self.draw_vector(x, y, z, dx, dy, dz, color='b')  # 使用蓝色箭头表示方向  
  #####################




        
    def clear_plot_elements(self):  
        # 遍历无人机列表，并从图上删除它们  
        for scatter in self.drone_plot_list:  
            scatter.remove()  
        self.drone_plot_list.clear() 
        for text in self.text_list:
            text.remove()

        self.text_list.clear() 
  
          # 清除速度向量  
        for line in self.vel_line_list:  
            line.remove()  # 注意这里假设vel_line_list包含的是Quiver或Line2D对象  
        self.vel_line_list.clear()  
  
        # 清除轨迹线  
        for line in self.trajectory_line_list:  
            # 如果trajectory_line_list包含的是Line2D对象的列表，你需要遍历每个列表  
            if isinstance(line, list):  
                for segment in line:  
                    segment.remove()  
            else:  
                line.remove()  
        self.trajectory_line_list.clear()  
  
        # 如果需要的话，强制重新绘制图形  
        #self.ax.figure.canvas.draw_idle()

    def animate(self):

        self.draw_drones()

        return self.ax.patches + self.ax.texts + self.ax.artists

    def show_ani(self):
        ani = animation.FuncAnimation(
        self.fig, self.animate, init_func=self.init_plot, interval=100, blit=True, frames=100, save_count=100)
        plt.show()
    
    def save_ani(self, name='animation'): 
        ani = animation.FuncAnimation(
        self.fig, self.animate, init_func=self.init_plot, interval=1, blit=False, save_count=300)
        ani.save(name+'.gif', writer='pillow')

    # # animation method 2
    def save_gif_figure(self, path, i, format='png'):

        if path.exists():
            order = str(i).zfill(3)
            plt.savefig(str(path)+'/'+order+'.'+format, format=format)
        else:
            path.mkdir()
            order = str(i).zfill(3)
            plt.savefig(str(path)+'/'+order+'.'+format, format=format)

    def create_animate(self, image_path, ani_path, ani_name='animated', keep_len=30, rm_fig_path=True):

        if not ani_path.exists():
            ani_path.mkdir()

        images = list(image_path.glob('*.png'))
        images.sort()
        image_list = []
        for i, file_name in enumerate(images):

            if i == 0:
                continue

            image_list.append(imageio.imread(file_name))
            if i == len(images) - 1:
                for j in range(keep_len):
                    image_list.append(imageio.imread(file_name))

        imageio.mimsave(str(ani_path)+'/'+ ani_name+'.gif', image_list)
        print('Create animation successfully')

        if rm_fig_path:
            shutil.rmtree(image_path)

    # old             
    def point_arrow_plot(self, point, length=0.5, width=0.3, color='r'):

        px = point[0, 0]
        py = point[1, 0]
        theta = point[2, 0]

        pdx = length * cos(theta)
        pdy = length * sin(theta)

        point_arrow = mpl.patches.Arrow(x=px, y=py, dx=pdx, dy=pdy, color=color, width=width)

        self.ax.add_patch(point_arrow)

    def point_list_arrow_plot(self, point_list=[], length=0.5, width=0.3, color='r'):

        for point in point_list:
            self.point_arrow_plot(point, length=length, width=width, color=color)

    
    def point_plot(self, point, markersize=2, color="k"):
        
        if isinstance(point, tuple):
            x = point[0]
            y = point[1]
        else:
            x = point[0,0]
            y = point[1,0]
    
        self.ax.plot([x], [y], marker='o', markersize=markersize, color=color)

    # plt 
    def cla(self):
        self.ax.cla()

    def pause(self, time=0.001):
        plt.pause(time)
    
    def show(self):
        plt.show()

    # def draw_start(self, start):
    #     self.ax.plot(start[0, 0], start[1, 0], 'rx')

    # def plot_trajectory(self, robot, num_estimator, label_name = [''], color_line=['b-']):

    #     self.ax.plot(robot.state_storage_x, robot.state_storage_y, 'g-', label='trajectory')

    #     for i in range(num_estimator):
    #         self.ax.plot(robot.estimator_storage_x[i], robot.estimator_storage_y[i], color_line[i], label = label_name[i])

    #     self.ax.legend()

    # def plot_pre_tra(self, pre_traj):
    #     list_x = []
    #     list_y = []

    #     if pre_traj != None:
    #         for pre in pre_traj:
    #             list_x.append(pre[0, 0])
    #             list_y.append(pre[1, 0])
            
    #         self.ax.plot(list_x, list_y, '-b')
    
    # def draw_path(self, path_x, path_y, line='g-'):
    #     self.ax.plot(path_x, path_y, 'g-')



    



