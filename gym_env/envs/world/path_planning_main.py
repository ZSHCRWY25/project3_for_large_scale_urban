'''
:@Author: 刘玉璞
:@Date: 2024/4/13 21:58:20
:@LastEditors: 刘玉璞
:@LastEditTime: 2024/7/2 16:44:58
:Description: 
'''
# 输入：地图，起止点
# 输出：航路（点集）+各航路航路点数量

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from envs.world.theta_star_3D import theta_star_3D
import math
def path_planning_main(sizeE,P0, Pend, E3d_safe):
    ##############################################################################################################

    x0 = math.ceil(P0[0])
    y0= math.ceil(P0[1])
    z0= math.ceil(P0[2])

    xend= math.ceil(Pend[0])
    yend = math.ceil(Pend[1])
    zend = math.ceil(Pend[2])
    d_grid = 1

    kg = 1
    kh = 1.25
    ke = np.sqrt((xend-x0)**2+(yend-y0)**2+(zend-z0)**2)


    ##############################################################################################################
    # Map definition

    ##############################################################################################################
    # Path generation

    # Store gains in vector
    K = [kg, kh, ke]

    # Start measuring execution time
    #start_time = time.time()
    #初始化轨迹

    # 生成轨迹

    [path, n_points] = theta_star_3D(K, E3d_safe, x0, y0, z0, xend, yend, zend, sizeE)


    # Stop measuring and print execution time

    #print(" %s seconds" % (time.time() - start_time))


    ##############################################################################################################

    X = np.arange(1, sizeE[0]-1, d_grid)
    Y = np.arange(1, sizeE[1]-1, d_grid)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.plot_surface(X, Y, E3d_safe[1:-1][:, 1:-1])
    ax.plot(path[:][:, 1], path[:][:, 0], path[:][:, 2], 'kx-')
    ax.plot([x0], [y0], [z0], 'go')
    ax.plot([xend], [yend], [zend], 'ro')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.auto_scale_xyz([0, sizeE[1]], [0, sizeE[0]], [0, sizeE[2]])
    plt.show()

    return path, n_points


