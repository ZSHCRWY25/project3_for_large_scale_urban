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
                
    def __init__(self, robot_class=mobile_robot, robot_number=0, robot_mode='omni', robot_init_mode = 0, step_time=0.1, components=[], **kwargs):

        self.robot_class = robot_class#分类
        self.robot_number = robot_number#数量
        self.init_mode = robot_init_mode#初始化模式
        self.robot_list = []
        self.cur_mode = robot_init_mode#当前模式
        self.com = components#存储组件列表

        self.interval = kwargs.get('interval', 1)#存储时间间隔，默认为 1
        self.square = kwargs.get('square', [0, 0, 10, 10] )#存储方形区域的坐标，默认为 [0, 0, 10, 10] -[x_min, y_min, x_max, y_max]
        self.circular = kwargs.get('circular', [5, 5, 4] )#存储圆形区域的坐标，默认为 [5, 5, 4]
        self.random_bear = kwargs.get('random_bear', False)#是否随机方向，默认为 False
        self.random_radius = kwargs.get('random_radius', False)#是否随机半径，默认为 False。


        # init_mode: 0 manually initialize
        #            1 single row
        #            2 random
        #            3 circular 
        # kwargs: random_bear random radius

        if self.robot_number > 0:
            if self.init_mode == 0:#如果 init_mode 为 0，要求关键字参数中包含 radius_list、init_state_list 和 goal_list。
                assert 'radius_list' and 'init_state_list' and 'goal_list' in kwargs.keys()
                radius_list = kwargs['radius_list']
                init_state_list = kwargs['init_state_list']
                goal_list = kwargs['goal_list']
            else:
                radius_list = kwargs.get('radius_list', [0.2])
                init_state_list, goal_list, radius_list = self.init_state_distribute(self.init_mode, radius=radius_list[0])
                #使用默认半径列表或从 init_state_distribute 函数中获取初始化状态和目标。

        # robot
        for i in range(self.robot_number):#遍历机器人数量，为每个机器人创建一个 robot 对象，并将其添加到 robot_list 中
            robot = self.robot_class(id=i, mode=robot_mode, radius=radius_list[i], init_state=init_state_list[i], goal=goal_list[i], step_time=step_time, **kwargs)
            self.robot_list.append(robot)
            self.robot = robot if i == 0 else None 
        
    def init_state_distribute(self, init_mode=1, radius=0.2):
        # init_mode: 1 single row
        #            2 random
        #            3 circular      
        # square area: x_min, y_min, x_max, y_max
        # circular area: x, y, radius
        
        num = self.robot_number
        state_list, goal_list = [], []

        if init_mode == 1:
             # single row
            state_list = [np.array([ [i * self.interval], [self.square[1]], [pi/2] ]) for i in range(int(self.square[0]), int(self.square[0])+num)]
            #i * self.interval，表示机器人的 x 坐标。self.square[1]，表示机器人的 y 坐标。pi/2，表示机器人的初始朝向（角度）
            goal_list = [np.array([ [i * self.interval], [self.square[3]] ]) for i in range(int(self.square[0]), int(self.square[0])+num)]
            goal_list.reverse()

        elif init_mode == 2:
            # random
            state_list, goal_list = self.random_start_goal()

        elif init_mode == 3:
            # circular
            circle_point = np.array(self.circular)#圆心坐标
            theta_step = 2*pi / num#表示每次迭代时角度的步进
            theta = 0#初始化 theta 为 0，表示从圆的起始位置开始

            while theta < 2*pi:
                state = circle_point + np.array([ cos(theta) * self.circular[2], sin(theta) * self.circular[2], theta + pi- self.circular[2] ])
                #cos(theta) * self.circular[2] 和 sin(theta) * self.circular[2] 计算机器人相对于圆心的 x 和 y 偏移量。
                #theta + pi - self.circular[2] 表示机器人的初始朝向（角度）
                #将这些偏移量添加到 circle_point，得到机器人的初始状态。
                goal = circle_point[0:2] + np.array([cos(theta+pi), sin(theta+pi)]) * self.circular[2]
                theta = theta + theta_step
                state_list.append(state[:, np.newaxis])
                goal_list.append(goal[:, np.newaxis])

        elif init_mode == 4:
            # random 2
            circle_point = np.array(self.circular)
            theta_step = 2*pi / num
            theta = 0

            while theta < 2*pi:
                state = circle_point + np.array([ cos(theta) * self.circular[2], sin(theta) * self.circular[2], theta + pi- self.circular[2] ])
                goal = circle_point[0:2] + np.array([cos(theta+pi), sin(theta+pi)]) * self.circular[2]
                theta = theta + theta_step
                state_list.append(state[:, np.newaxis])
                goal_list.append(goal[:, np.newaxis])

        elif init_mode == 5:
            
            half_num = int(num /2)

            state_list1 = [np.array([ [i * self.interval], [self.square[1]], [pi/2] ]) for i in range(int(self.square[0]), int(self.square[0])+half_num)]

            state_list2 = [np.array([ [i * self.interval], [self.square[3]], [pi/2] ]) for i in range(int(self.square[0]), int(self.square[0])+half_num)]
            state_list2.reverse()
            
            goal_list1 = [np.array([ [i * self.interval], [self.square[3]], [pi/2] ]) for i in range(int(self.square[0]), int(self.square[0])+half_num)]
            goal_list1.reverse()

            goal_list2 = [np.array([ [i * self.interval], [self.square[1]], [pi/2] ]) for i in range(int(self.square[0]), int(self.square[0])+half_num)]
            
            state_list, goal_list = state_list1+state_list2, goal_list1+goal_list2
                    
        if self.random_bear:
            for state in state_list:
                state[2, 0] = np.random.uniform(low = -pi, high = pi)

        if self.random_radius:
            radius_list = np.random.uniform(low = 0.2, high = 1, size = (num,))
        else:
            radius_list = [radius for i in range(num)]

        return state_list, goal_list, radius_list
    
    def random_start_goal(self):

        num = self.robot_number
        random_list = []
        goal_list = []
        while len(random_list) < 2*num:

            new_point = np.random.uniform(low = self.square[0:2]+[-pi], high = self.square[2:4]+[pi], size = (1, 3)).T

            if not self.check_collision(new_point, random_list, self.com, self.interval):
                random_list.append(new_point)

        start_list = random_list[0 : num]
        goal_temp_list = random_list[num : 2 * num]

        for goal in goal_temp_list:
            goal_list.append(np.delete(goal, 2, 0))

        return start_list, goal_list
    
    def random_goal(self):

        num = self.robot_number
        random_list = []
        goal_list = []
        while len(random_list) < num:

            new_point = np.random.uniform(low = self.square[0:2]+[-pi], high = self.square[2:4]+[pi], size = (1, 3)).T

            if not self.check_collision(new_point, random_list, self.com, self.interval):
                random_list.append(new_point)

        goal_temp_list = random_list[:]

        for goal in goal_temp_list:
            goal_list.append(np.delete(goal, 2, 0))

        return goal_list

    def distance(self, point1, point2):
        diff = point2[0:2] - point1[0:2]
        return np.linalg.norm(diff)

    def check_collision(self, check_point, point_list, components, range):

        circle = namedtuple('circle', 'x y r')
        point = namedtuple('point', 'x y')
        self_circle = circle(check_point[0, 0], check_point[1, 0], range/2)

        for obs_cir in components['obs_circles'].obs_cir_list:
            temp_circle = circle(obs_cir.state[0, 0], obs_cir.state[1, 0], obs_cir.radius)
            if collision_cir_cir(self_circle, temp_circle):
                return True
        
        # check collision with map
        if collision_cir_matrix(self_circle, components['map_matrix'], components['xy_reso'], components['offset']):
            return True

        # check collision with line obstacles
        for line in components['obs_lines'].obs_line_states:
            segment = [point(line[0], line[1]), point(line[2], line[3])]
            if collision_cir_seg(self_circle, segment):
                return True

        for point in point_list:
            if self.distance(check_point, point) < range:
                return True
                
        return False


    def step(self, vel_list=[], **vel_kwargs):

        # vel_kwargs: vel_type = 'diff', 'omni'
        #             stop=True, whether stop when arrive at the goal
        #             noise=False, 
        #             alpha = [0.01, 0, 0, 0.01, 0, 0], noise for diff
        #             control_std = [0.01, 0.01], noise for omni

        for robot, vel in zip(self.robot_list, vel_list):
            robot.move_forward(vel, **vel_kwargs)

    def cal_des_list(self):
        vel_list = list(map(lambda x: x.cal_des_vel() , self.robot_list))
        return vel_list
    
    def cal_des_omni_list(self):
        vel_list = list(map(lambda x: x.cal_des_vel_omni() , self.robot_list))
        return vel_list

    def arrive_all(self):

        for robot in self.robot_list:
            if not robot.arrive():
                return False

        return True

    def robots_reset(self, reset_mode=1, **kwargs):
        
        if reset_mode == 0:
            for robot in self.robot_list:
                robot.reset(self.random_bear)
        
        elif self.cur_mode != reset_mode:
            state_list, goal_list, _ = self.init_state_distribute(init_mode = reset_mode)

            for i in range(self.robot_number):
                self.robot_list[i].init_state = state_list[i]
                self.robot_list[i].goal = goal_list[i]
                self.robot_list[i].reset(self.random_bear) 
            
            self.cur_mode = reset_mode

        elif reset_mode == 2:
            state_list, goal_list = self.random_start_goal()
            for i in range(self.robot_number):
                self.robot_list[i].init_state = state_list[i]
                self.robot_list[i].goal = goal_list[i]
                self.robot_list[i].reset(self.random_bear) 
        
        elif reset_mode == 4:
            goal_list = self.random_goal()
            for i in range(self.robot_number):
                self.robot_list[i].goal = goal_list[i]
                self.robot_list[i].reset(self.random_bear)

        else:
            for robot in self.robot_list:
                robot.reset(self.random_bear)

    def robot_reset(self, id=0):
        self.robot_list[id].reset(self.random_bear)

    def total_states(self):
        robot_state_list = list(map(lambda r: np.squeeze( r.omni_state()), self.robot_list))
        nei_state_list = list(map(lambda r: np.squeeze( r.omni_obs_state()), self.robot_list))
        obs_circular_list = list(map(lambda o: np.squeeze( o.omni_obs_state() ), self.com['obs_circles'].obs_cir_list))
        obs_line_list = self.com['obs_lines'].obs_line_states

        return [robot_state_list, nei_state_list, obs_circular_list, obs_line_list]
    # # states
    @staticmethod
    def distance(point1, point2):
        return math.sqrt( (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2 )