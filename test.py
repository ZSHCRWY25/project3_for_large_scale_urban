
import matplotlib.pyplot as plt  
from matplotlib.animation import FuncAnimation  
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib.patches import FancyArrow  
import numpy as np  

fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')  

previous_state = [0,0,0]
state = [5,5,5]
trajectory_line_list = []

x_list = [previous_state[0], state[0]]  
y_list = [previous_state[1], state[1]]  
z_list = [previous_state[2], state[2]]  
c = ax.plot(x_list, y_list, z_list, color = 'r')[0]
trajectory_line_list.append(c)

for line in trajectory_line_list:  
    line.remove()  # 直接从列表中获取Line3D对象并删除  
trajectory_line_list.clear()
plt.show()



#以下是成功箭头
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# x,y,z = 5,5,5
# dx,dy,dz =2,1,3 
# # 绘制箭头
# ax.quiver(x,y,z,dx,dy,dz)
# # 设置华标釉范围
# ax.set_xlim([0,10])
# ax.set_ylim([0,10])
# ax.set_zlim([0,10])
# # 没置些标釉标备
# ax.set_xlabel('X')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()

#以下是一段成功的轨迹
# class FancyArrow3D(FancyArrow):  
#     def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):  
#         # 这里只是模拟，实际实现需要3D转换  
#         super().__init__((0, 0), (1, 1), *args, **kwargs)  
#         # 注意：这只是一个占位符，不会在3D中显示  
  
# class Drone:  
#     def __init__(self):  
#         # 模拟一些航点数据  
#         self.waypoints = np.array([  
#             [0, 0, 0],  # 起点  
#             [1, 1, 1],  
#             [2, 2, 2],  
#             [3, 0, 3],  # 中间航点（用于测试draw_waypoints）  
#             [4, 4, 4],  # 终点  
#             # 假设最后一个元素包含方向信息（这里只是示例，方向通常不会与航点一起存储）  
#             [4.1, 4.2, 4.3]    
#         ])  
  
# class Visualizer:  
#     def __init__(self):  
#         self.fig = plt.figure()  
#         self.ax = self.fig.add_subplot(111, projection='3d')  
  
#     def draw_vector(self, x, y, z, dx, dy, dz, color='r'):  
#         # ... 使用FancyArrow3D的模拟实现 ...  
#         a = FancyArrow3D(x, y, z, dx, dy, dz, mutation_scale=20, lw=1, arrowstyle="-|>", color=color)  
#         self.ax.add_artist(a)  
  
#     def draw_trajectory(self, traj, color='g', label='trajectory', show_direction=False, refresh=False):  
#         path_x_list = traj[:, 0]  
#         path_y_list = traj[:, 1]  
#         path_z_list = traj[:, 2]  
  
#         # 绘制轨迹线  
#         line = self.ax.plot(path_x_list, path_y_list, path_z_list, color=color, label=label)  
  
#         if show_direction:  
#             # 假设traj的最后一个元素包含了方向（单位向量），我们需要先计算方向向量的长度  
#             if traj.shape[1] > 3:  
#                 # 假设最后三个元素是dx, dy, dz（方向向量的分量）  
#                 dx_list = traj[:, -3] / np.linalg.norm(traj[:, -3:], axis=1, keepdims=True)  
#                 dy_list = traj[:, -2] / np.linalg.norm(traj[:, -3:], axis=1, keepdims=True)  
#                 dz_list = traj[:, -1] / np.linalg.norm(traj[:, -3:], axis=1, keepdims=True)  
  
#                 # 在每个点上绘制方向箭头  
#                 for i in range(len(traj)):  
#                     x, y, z = traj[i, :3]  # 提取点的坐标  
#                     dx, dy, dz = dx_list[i], dy_list[i], dz_list[i]  # 提取方向向量的分量  
#                     self.draw_vector(x, y, z, dx, dy, dz, color='b')  # 使用蓝色箭头表示方向  
  
#         if refresh:  
#             # 在这里，你可能想保存line对象以便后续操作，但通常matplotlib不需要这样做来更新图形  
#             # 因为你可以通过重新调用draw_trajectory来更新图形  
#             pass 
#         # 设置坐标轴标签和标题（可选）  
#         self.ax.set_xlabel('X')  
#         self.ax.set_ylabel('Y')  
#         self.ax.set_zlabel('Z')  
#         self.ax.set_title('3D Trajectory')  
  
#         # 显示图例（可选）  
#         self.ax.legend()  
  
#         # 显示图形  
#         plt.show()
  
  
#     def draw_waypoints(self, drone):  
#         middle_waypoints = drone.waypoints[1:-1]    
#         x = [wp[0] for wp in middle_waypoints]  
#         y = [wp[1] for wp in middle_waypoints]  
#         z = [wp[2] for wp in middle_waypoints]  
#         self.ax.scatter(x, y, z, c='r',marker='o')
#         for i in range(len(x)):  
#             self.ax.text(x[i],y[i],z[i],str(i), zdir='y')
#         # ... 您的draw_waypoints的其余部分 ...  
  
#         # 显示图形  
#         plt.show()  
  
# # 使用示例  
# drone = Drone()  
# viz = Visualizer()  
# viz.draw_trajectory(traj = drone.waypoints, color='b', show_direction=True)  # 忽略方向信息的最后一个元素  
# viz.draw_waypoints(drone)



# def draw_waypoints(waypoints):
#     middle_waypoints = waypoints[1:-1]  
 
#     x = [wp[0] for wp in middle_waypoints]  
#     y = [wp[1] for wp in middle_waypoints]  
#     z = [wp[2] for wp in middle_waypoints] 
#     ax = plt.figure().add_subplot(111,projection='3d')

#     ax.scatter(x, y, z,c='r',marker='o')
#     for i in range(len(x)):
#        ax.text(x[i],y[i],z[i],str(i), zdir='y')



# waypoints = [[3,6,8],
#              [7,3,9],
#              [8,2,6],
#              [8,4,6],
#              [9,4,7]]

# draw_waypoints(waypoints)
