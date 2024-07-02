'''
:@Author: 刘玉璞
:@Date: 2024/4/13 20:48:51
:@LastEditors: 刘玉璞
:@LastEditTime: 2024/7/2 16:44:46
:Description: 
'''
#生成位置、高度、半径随机的圆柱、
#输入：起点终点，地图大小、起始点周围空白区域大小n_low
#输出：E二维障碍物地图, E_safe二维障碍物带保护区地图, 三维：E3d, E3d_safe；obs_list障碍物坐标、中心点、半径
import math
import numpy as np
from numpy.random import default_rng
import pandas as pd
#初始化
P0 = [0,0,0]
Pend = [0,0,0]

def grid_3D_safe_zone(sizeE, d_grid,h, path_num, f1origin, f2origin, n_low):
    #初始化

    y_size = sizeE[0]
    x_size = sizeE[1]
    z_size = sizeE[2]

    z_grid = np.linspace(1, z_size, z_size//d_grid)


    ##############################################################################################################
    # 初始化障碍物参数
    mean_E = 0
    sigma = 1
    k_sigma = 3   # 障碍物密度
    rng = default_rng()
    E = rng.normal(mean_E, sigma, (y_size, x_size))
    sigma_obstacle = k_sigma * sigma
    #二维地图，存储的是障碍物中心位置
    E = np.uint8(E > sigma_obstacle)


    #最小高度
    hh_min = 4

    # 先复制一个二维地图，一会要用
    EE = E.copy()
    #初始化障碍物数量
    obs_num=0
    obs_list=[]

    for i in range(y_size):
        for j in range(x_size):
            # 检查栅格值
            if EE[j, i] > 0:
                # 指定障碍物高度
                hh = round(rng.normal(0.8*h, 0.5*h))
                # 限定高度
                if hh < hh_min:
                    hh = hh_min
                elif hh > z_size:
                    hh = z_size
                E[j, i] = hh


    #清空起始点周围
    for i in range(path_num):
            P0[0] = math.ceil(f1origin[i][0])
            P0[1] = math.ceil(f1origin[i][1])
            P0[2] = math.ceil(f1origin[i][2])

            Pend[0] = math.ceil(f2origin[i][0])
            Pend[1] = math.ceil(f2origin[i][1])
            Pend[2] = math.ceil(f2origin[i][2])

            E[np.ix_(np.arange(P0[0] - n_low, P0[0] + n_low + 1), np.arange(P0[1] - n_low, P0[1] + n_low + 1))] = 0
            E[np.ix_(np.arange(Pend[0] - n_low, Pend[0] + n_low + 1), np.arange(Pend[1] - n_low, Pend[1] + n_low + 1))] = 0

    #三维地图
    E3d = np.zeros((y_size, x_size, z_size))

    #把占用栅格标为1
    for i in range(z_size):
        E3d[np.ix_(np.arange(0, y_size), np.arange(0, x_size), [i])] = np.atleast_3d(E[0:y_size][:, 0:x_size] >= z_grid[i])


    ##############################################################################################################
    # 为建筑物生成半径
    E_safe = E.copy()

    for i in range(x_size):
        for j in range(y_size):
            if EE[j,i]>0:
                obs_num += 1
                obs_radius = np.random.randint(2,5)#半径2-4
                obs_list.append([j, i,E[j,i] ,obs_radius])
                for k in range(i - obs_radius, i + obs_radius + 1):
                    if k < 1 or k >= x_size:
                        break
                    else:
                        for l in range(j - obs_radius, j + obs_radius + 1):
                            if l < 1 or l >= y_size:
                                break
                            else:
                                if E_safe[l, k] < E[j, i]:
                                    E_safe[l, k] = E[j, i]
                                    for n in range(E[j, i]):
                                        E3d[l, k, n] = 1

    ##############################################################################################################

    E3d_safe = E3d.copy()

    for i in range(x_size):
        for j in range(y_size):
            for k in range(z_size):

                # Check neighbour nodes
                l = np.arange(i - 1, i + 2)
                m = np.arange(j - 1, j + 2)
                n = np.arange(k - 1, k + 2)

                # Limit neighbours within the grid
                if min(l) < 0:
                    l = np.arange(i, i + 2)
                elif max(l) >= x_size:
                    l = np.arange(i - 1, i + 1)
                if min(m) < 0:
                    m = np.arange(j, j + 2)
                elif max(m) >= y_size:
                    m = np.arange(j - 1, j + 1)
                if min(n) < 0:
                    n = np.arange(k, k + 2)
                elif max(n) >= z_size:
                    n = np.arange(k - 1, k + 1)

                E_eval = E3d[m][:, l][:, :, n]

                if E3d[j, i, k] == 0 and E_eval.max() == 1:
                    # 安全区是0.5
                    E3d_safe[j, i, k] = 0.5


    ##############################################################################################################
    #电子围栏

    E[np.ix_([0, -1], np.arange(0, x_size))] = z_size
    E[np.ix_(np.arange(0, x_size), [0, -1])] = z_size

    E_safe[np.ix_([0, -1], np.arange(0, y_size-1))] = z_size
    E_safe[np.ix_(np.arange(0, x_size), [0, -1])] = z_size

    E3d[np.ix_([0, -1], np.arange(0, y_size), np.arange(0, z_size))] = 1
    E3d[np.ix_(np.arange(0, x_size), [0, -1], np.arange(0, z_size))] = 1
    E3d[np.ix_(np.arange(0, x_size), np.arange(0, y_size), [0, -1])] = 1

    E3d_safe[np.ix_([0, -1], np.arange(0, y_size), np.arange(0, z_size))] = 1
    E3d_safe[np.ix_(np.arange(0, x_size), [0, -1], np.arange(0, z_size))] = 1
    E3d_safe[np.ix_(np.arange(0, x_size), np.arange(0, y_size), [0, -1])] = 1


    return E, E_safe, E3d, E3d_safe, obs_list

##测试一下
#x_size,y_size,z_size = sizeE = [200, 200, 20]
#d_grid = 1
#_low = 3
#path_num = 2
#f1origin = [[5,5,5],[1,1,1]]#起始位置列表
#f2origin = [[10,10,10],[3,3,3]]#后面可以从表里读
#third_column = np.array(f2origin)[:, 2]
#h = np.max(third_column)
#初始化
#obs_list = []
#paths = []
#生成地图

#[E, E_safe, E3d, E3d_safe ,obs_list] = grid_3D_safe_zone(sizeE, d_grid, h, path_num, f1origin, f2origin, n_low)

#print(obs_list)