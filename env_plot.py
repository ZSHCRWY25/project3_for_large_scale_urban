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
from matplotlib.patches import FancyArrow3D  
from math import cos, sin, pi
from pathlib import Path
import inspect
import matplotlib.transforms as mtransforms
from matplotlib.patches import Polygon

class env_plot:
    def __init__(self, map_size, building_list, lenth= 10, width=10, height=10, components=dict(),  
                 full=False, keep_path=False, map_matrix=None, offset_x = 0, offset_y=0, **kwargs):

        self.fig = plt.figure()  
        self.ax = self.fig.add_subplot(111, projection='3d')  
        self.drone_plot_list = []
        
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
        self.vel_line_list = []

        self.init_plot(**kwargs)

        if full:#full为True，则尝试将图形窗口设置为全屏模式（根据操作系统）。
            mode = platform.system()
            if mode == 'Linux':
                plt.get_current_fig_manager().full_screen_toggle()
            elif mode == 'Windows':
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()

    def init_plot(self, **kwargs):##没改
        self.ax.set_aspect('equal')#确保x和y轴的比例相
        self.ax.set_xlim(self.offset_x, self.offset_x + self.width)#设置x和y轴的限制，使用offset_x和offset_y进行偏移
        self.ax.set_ylim(self.offset_y, self.offset_y + self.height)
        self.ax.set_xlabel("x [m]")#设置x和y轴的标签
        self.ax.set_ylabel("y [m]")

        current_file_frame = inspect.getfile(inspect.currentframe())
        car_image_path = Path(current_file_frame).parent / 'car0.png'
        self.init_car_img = image.imread(str(car_image_path)) #加载汽车图像

        self.plot_buildings_on_map(self.map_size, self.building_list)#绘制环境的静态组件
        drones = self.components.get('robots', None)
        for drone in drones.Drone_list_list:
            self.draw_waypoints(drone)    
        return self.ax.patches + self.ax.texts + self.ax.artists


    def plot_buildings_on_map(map_size, buildings):  
        # 地图大小可能包含x、y和z轴的范围，但在这里我们只关心x和y的范围  
        x_range, y_range = map_size  
        
        fig = plt.figure(figsize=(8, 6))  
        ax = fig.add_subplot(111, projection='3d')  
      
        # 设置地图的x和y轴范围  
        ax.set_xlim(-x_range/2, x_range/2)  
        ax.set_ylim(-y_range/2, y_range/2)  
        ax.set_zlim(0, max(b[2] for b in buildings) + 0.1)  # z轴范围基于建筑物最高度+一点额外空间  
      
        # 绘制每个建筑物（圆柱体）  
        for b in buildings:  
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
      
            # 添加坐标轴标签  
            ax.set_xlabel('X')  
            ax.set_ylabel('Y')  
            ax.set_zlabel('Z')  


    def draw_drones(self):#改完
        drones = self.components.get('robots', None)
        for drone in drones.Drone_list_list:
            self.draw_drone(drone)



    def draw_drone(self, drone, drone_color = 'g', destination_color='r', show_vel=True, show_goal=True, show_text=True, show_traj=False, traj_color='-g', **kwargs):
        
        x = drone.state[0]
        y = drone.state[1]
        z = drone.state[2]
        
        goal_x = drone.destination[0]
        goal_y = drone.destination[1]
        goal_z = drone.destination[2]

        drone_sphere = plt.sphere((x, y, z), drone.radius, color=drone_color) 
        self.ax.scatter([x], [y], [z], color=drone_color)  

        goal_sphere = plt.sphere((goal_x, goal_y, goal_z), drone.radius, color=destination_color, alpha=0.5)  # 同样使用球体代替圆  
        self.ax.scatter([goal_x], [goal_y], [goal_z], color=destination_color, alpha=0.5)  
  
        
        if show_goal:  
            if show_text:  
                self.ax.text(goal_x + 0.3, goal_y, goal_z, 'G' + str(drone.id), fontsize=12, color='k')  
            # 这里不将球体添加到drone_plot_list，因为它们是使用scatter绘制的点


        if show_text:  
            self.ax.text(x - 0.5, y, z, 'D' + str(drone.id), fontsize=10, color='k')  
  
        if show_traj:  
            x_list = [drone.previous_state[0], drone.state[0]]  
            y_list = [drone.previous_state[1], drone.state[1]]  
            z_list = [drone.previous_state[2], drone.state[2]]  
            self.ax.plot(x_list, y_list, z_list, color = traj_color)  
  
        if show_vel:  
            vel_x, vel_y, vel_z = drone.state[3:]  
            # 使用FancyArrow3D绘制3D箭头  
            a = self.draw_vector(self, x, y, z, vel_x, vel_y, vel_z, color='r') 
            self.ax.add_artist(a)  
            self.drone_plot_list.append(a)

         # 更新绘图  
        plt.draw()

    def draw_vector(self, x, y, z, vel_x, vel_y, vel_z, color='r'):  
        vel_tt = vel_x + vel_y + vel_z
        dx,dy,dz =x + round(vel_x/vel_tt,1), y + round(vel_y/vel_tt,1), z + round(vel_z/vel_tt,1)
        # 绘制箭头
        self.ax.quiver(x,y,z,dx,dy,dz)

    def draw_trajectory(self, traj, color='g', label='trajectory', show_direction=False, refresh=False):  
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
  
        plt.show()

    def draw_waypoints(drone):
        middle_waypoints = drone.waypoints[1:-1]  
 
        x = [wp[0] for wp in middle_waypoints]  
        y = [wp[1] for wp in middle_waypoints]  
        z = [wp[2] for wp in middle_waypoints] 
        ax = plt.figure().add_subplot(111,projection='3d')

        ax.scatter(x, y, z,c='r',marker='o')
        for i in range(len(x)):
            ax.text(x[i],y[i],z[i],str(i), zdir='y')



        
    def clear_plot_elements(self):  
        # 清理文本  
        for text in self.ax.texts:  
            text.remove()  
  
        # 清理机器人图形  
        for robot_plot in self.robot_plot_list:  
            robot_plot.remove()  
  
        # 清理汽车图形  
        for car_plot in self.car_plot_list:  
            car_plot.remove()  
  
        # 清理线条（假设line_list是一个包含Line2D对象的列表）  
        for line in self.line_list:  
            line.remove()  
  
        # 清理激光雷达线条（类似地处理）  
        for lidar_line in self.lidar_line_list:  
            lidar_line.remove()  
  
     # 清理汽车图像  
        for car_img in self.car_img_show_list:  
            car_img.remove()  
  
        # 清理动态障碍物图形  
        for obs in self.dyna_obs_plot_list:  
            obs.remove()  
  
        # 清空列表，以便下次使用  
        self.car_plot_list.clear()  
        self.robot_plot_list.clear()  
        self.lidar_line_list.clear()  
        self.car_img_show_list.clear()  
        self.line_list.clear()  
        self.dyna_obs_plot_list.clear()
        # animation method 1


    def animate(self):

        self.draw_robot_diff_list()

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



    



