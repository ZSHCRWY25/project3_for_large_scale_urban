import numpy as np
import math
from line_sight_partial_3D import line_sight_partial_3D as lsp
class Drone:
    def __init__(self, id, starting, destination, waypoints, n_points,x,y,z, vx, vy, vz, ax, ay, az, dt):

        self.id = id#无人机编号

        self.starting = starting
        self.destination = destination#起止点

        self.waypoints = waypoints
        self.n_points = n_points#航路信息（个数、途经航路点）

        self.i = 1##初始化
        self.current_des = waypoints[self.i]

        self.px = x#位置
        self.py = y
        self.pz = z

        self.vx = vx#速度
        self.vy = vy
        self.vz = vz

        self.ax = ax#加速度
        self.ay = ay
        self.az = az

        self.dt = dt#时间步长

        self.arriveflag = False

    def v_new(self):

        self.vx += self.ax * self.dt
        self.vy += self.ay * self.dt
        self.vz += self.az * self.dt

    def p_new(self):
        self.px += self.vx * self.dt
        self.py += self.vy * self.dt
        self.pz += self.vz * self.dt

    def current_des_new(self,E3d, map_size):
        P = [self.px, self.py ,self.pz]

        if Drone.distance(P,self.current_des) <= 2:
            if self.i < self.n_points:
                x1,y1,z1 = self.waypoints[(self.i+1)]
                if lsp(E3d,(self.px, x1),(self.py, y1),(self.pz, z1),map_size) == 1:
                    self.i +=1 
                    self.current_des = self.waypoints[self.i]
            else:
                self.arriveflag = True

    @staticmethod
    def distance(point1, point2):
        return math.sqrt( (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2 )