import numpy as np
from math import sqrt, pi, cos, sin

# point: np(2,)
# segment: [point1, point2]
# circle: np(2,)
# r: 1

def range_seg_matrix(segment, matrix, reso, point_step_weight=2, offset=np.zeros(2,)):#用于计算线段与二进制矩阵（网格）之间的交互

    if matrix is None:#输入的矩阵是否为空
        return False, None, None

    init_point = segment[0]
    diff = segment[1] - segment[0]#提取线段的起点和终点，计算线段的长度
    len_seg = np.linalg.norm(diff)

    slope = diff / len_seg#线段的斜率（slope

    point_step = point_step_weight*reso#根据输入的 point_step_weight 和分辨率 reso，确定采样点的步长（point_step）
    cur_len = 0#起点开始，沿着线段方向采样点，计算每个采样点的位置

    while cur_len <= len_seg:
 
        cur_point = init_point + cur_len * slope

        cur_len = cur_len + point_step
        

        index = (cur_point - offset) / reso#采样点的位置转换为矩阵索引（index

        if index[0] < 0 or index[0] > matrix.shape[0] or index[1] < 0 or index[1] > matrix.shape[1]:#查索引是否越界，如果越界，计算采样点到起点的距离（lrange），并返回 True，表示发生碰撞
            lrange = np.linalg.norm( cur_point -  init_point)
            return True, cur_point, lrange

        elif matrix[int(index[0]), int(index[1])]:#如果索引未越界，检查矩阵中对应的单元格是否设置（非零）。如果设置，同样计算距离并返回 True
            lrange = np.linalg.norm( cur_point -  init_point)

            return True, cur_point, lrange#如果遍历完整个线段都没有发生碰撞，返回 False，表示没有交互

    return False, None, None


def range_cir_seg(circle, r, segment,):#于计算圆形障碍物与线段之间的交互
    # reference: https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm

    sp = segment[0]#提取线段的起点和终点
    ep = segment[1]

    d = ep - sp
    f = sp - circle#计算线段的方向向量 d 和起点到圆心的向量 f

    a = d @ d
    b = 2* f@d
    c = f@f - r ** 2

    discriminant = b**2 - 4 * a * c#二次方程的判别式计算交点个数

    if discriminant < 0:
        return False, None, None
    
    else:
        t1 = (-b - sqrt(discriminant)) / (2*a)
        t2 = (-b + sqrt(discriminant)) / (2*a)

        # 3x HIT cases:
        #          -o->             --|-->  |            |  --|->
        # Impale(t1 hit,t2 hit), Poke(t1 hit,t2>1), ExitWound(t1<0, t2 hit), 

        # 3x MISS cases:
        #       ->  o                     o ->              | -> |
        # FallShort (t1>1,t2>1), Past (t1<0,t2<0), CompletelyInside(t1<0, t2>1)

        if t1 >=0 and t1<=1:
            int_point = sp + t1 * d
            lrange = np.linalg.norm(int_point - sp)

            return True, int_point, lrange

        # if t2 >= 0 and t2 <=1:
           #  ExitWound(t1<0, t2 hit), 
        return False, None, None

    
def range_seg_seg(segment1, segment2):#用于计算两条线段之间的交互。
    # reference https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    # p, p+r, q, q+s

    p = segment1[0]#，提取两条线段的起点和方向向量
    r = segment1[1] - segment1[0]
    q = segment2[0]
    s = segment2[1] - segment2[0]

    temp1 = np.cross(r, s)#两个向量的叉乘结果
    temp2 = np.cross(q-p, r)

    if temp1 == 0 and temp2 == 0:#叉乘结果判断是否平行
        # collinear
        t0 = (q-p) @ r / (r @ r)
        t1 = t0 + s @ r / (r @ r)

        if max(t0, t1) >= 0 and min(t0, t1) < 0:#平行且不相交，返回 False。否则，计算参数 t 和 u，判断是否存在交点
            int_point = p
            lrange = 0
            return True, int_point, lrange
        
        elif min(t0, t1) >=0 and min(t0, t1) <= 1:
            int_point = p + min(t0, t1) * r
            lrange = np.linalg.norm(int_point - p)
            return True, int_point, lrange

        else:
            return False, None, None
    
    elif temp1 == 0 and temp2 != 0:
        # parallel and non-intersecting
        return False, None, None

    elif temp1 != 0:

        t = np.cross( q-p, s) / np.cross(r, s)
        u = np.cross( q-p, r) / np.cross(r, s)

        if t >=0 and t<=1 and u>=0 and u<= 1:

            int_point = p + t*r
            lrange = np.linalg.norm(int_point - p)
            return True, int_point, lrange
        else: 
            return False, None, None

    else:
        # not parallel and not intersect
        return False, None, None#返回交点、距离和是否发生交互的信息



