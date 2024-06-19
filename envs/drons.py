import numpy as np
from math import sin, cos, atan2, pi, sqrt
from line_sight_partial_3D import line_sight_partial_3D as lsp
from collections import namedtuple
class Drone:
    def __init__(self, id, starting, destination, waypoints, n_points, init_acc, dt=1, init_state=np.zeros((3, 1)),  
                 vel = np.zeros((3, 1)), vel_max = 2* np.ones((3, 1)), goal = np.zeros((2, 1)), goal_threshold = 0.1, radius = 0.2, *kwargs):

        self.id = int(id)#无人机编号
        if isinstance(init_state, list): 
            init_state = np.array(init_state, ndmin=3).T


        if isinstance(vel, list): 
            vel = np.array(vel, ndmin=3).T

        if isinstance(vel_max, list): 
            vel_max = np.array(vel_max, ndmin=3).T 

        if isinstance(goal, list): 
            goal = np.array(goal, ndmin=3).T

        self.state = init_state#机器人的初始位置
        self.previous_state = init_state

        self.vel = vel#全向模式下的速度控制。
        self.vel_max = vel_max#速度的最大限制

        self.starting = starting
        self.destination = destination#起止点

        self.waypoints = waypoints
        self.n_points = n_points#航路信息（个数、途经航路点）

        self.i = 1##初始化
        self.current_des = waypoints[self.i]

        self.acc = init_acc#加速度

        self.dt = dt#时间步长

        self.arrive_flag = False#到达标志
        self.collision_flag = False#碰撞标准
        self.goal_threshold = goal_threshold
        self.radius_collision = round(radius + kwargs.get('radius_exp', 0.1), 2)
        self.__noise = kwargs.get('noise', False)
        self.__alpha = kwargs.get('alpha', [0.03, 0, 0, 0.03, 0, 0])
        self.__control_std = kwargs.get('control_std', [0.01, 0.01])

    def update_info(self, state, vel):
    # 更新状态
        self.state = state
        self.vel = vel

    def move_forward(self, vel, stop=True, **kwargs):

        # default: robot mode: diff, no noise, vel_type: diff
        # vel_type: diff: np.array([[linear], [angular]])
        #           omni: np.array([[x], [y]])
        # kwargs: guarantee_time = 0.2, tolerance = 0.1, mini_speed=0.02, 
    
        if isinstance(vel, list):
            vel = np.array(vel, ndmin=2).T

        if vel.shape == (3,):
            vel = vel[:,np.newaxis]

        assert vel.shape == (3, 1)

        vel = np.clip(vel, -self.vel_max, self.vel_max)#限制范围
        
        if stop:
            if self.arrive_flag or self.collision_flag:
                vel = np.zeros((2, 1))


        self.previous_state = self.state
        self.move(vel, self.__noise, self.__control_std)

        self.arrive()


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


    def move(self, vel, noise, std):
        # vel_omni: np.array([[vx], [vy]])
        next_state = Drone.motion(self.state, vel, self.dt, noise, std)
        
        self.state = next_state
        self.vel = vel


    def motion(current_state, vel, sampletime, noise = False, control_std = [0.01, 0.01, 0.01]):

    # vel: np.array([[vel x], [vel y]])
    # current_state: np.array([[x], [y]])

        if noise == True:
            vel_noise = vel + np.random.normal([[0], [0], [0]], scale = [[control_std[0]], [control_std[1]], [control_std[1]]])
        else:
            vel_noise = vel

        next_state = current_state + vel_noise * sampletime
    
        return next_state
    
    def arrive(self):

        position = self.state[0:3]
        dist = np.linalg.norm(position - self.goal[0:3]) 

        if dist < self.goal_threshold:
            self.arrive_flag = True
            return True
        else:
            self.arrive_flag = False
            return False
    
    @staticmethod
    def distance(point1, point2):
        distance = sqrt((point1[0]-point1[0])**2+(point1[0]-point1[1])**2+(point1[1]-point1[1])**2)
        return distance             
   