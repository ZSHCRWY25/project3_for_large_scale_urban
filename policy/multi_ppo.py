'''
:@Author: 刘玉璞
:@Date: 2024/6/22 16:45:09
:@LastEditors: 刘玉璞
:@LastEditTime: 2024/7/2 16:45:09
:Description: 
'''
import numpy as np
import torch
from torch.optim import Adam
from pathlib import Path
import scipy
import scipy.signal
import time
import os
from train.post_train import post_train
import threading
from mpi4py import MPI

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)#用于组合一个长度（length）和一个形状（shape）参数，以返回一个元组，该元组表示一个多维数组的形状

def discount_cumsum(x, discount):##计算折扣因子
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class multi_PPObuf:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):#

        # gamma: discount factor
        # Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)

        self.obs_buf = [0] * size#观测
        #size表示缓冲区可以存储的最大交互步数
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)#动作
        self.adv_buf = np.zeros(size, dtype=np.float32)#优势
        self.rew_buf = np.zeros(size, dtype=np.float32)#奖励
        self.ret_buf = np.zeros(size, dtype=np.float32)#回报
        self.val_buf = np.zeros(size, dtype=np.float32)#价值
        self.logp_buf = np.zeros(size, dtype=np.float32)#日志概率
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
    
    def store(self, obs, act, rew, val, logp):#将一步的代理-环境交互数据存储到缓冲区
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        #观测、动作、奖励、价值和日志概率。
        self.obs_buf[self.ptr] = obs.copy()
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1#指针

    def finish_path(self, last_val=0):#当一个完整的交互路径（比如一个完整的回合或轨迹）完成时，这个方法被调用
        #last_val表示在路径结束时估计的价值

        path_slice = slice(self.path_start_idx, self.ptr)##将路径进行切片
        rews = np.append(self.rew_buf[path_slice], last_val)#将切片后的路径与奖励链接起来，形成完整的奖励序列。这包括路径结束时的估计价值
        vals = np.append(self.val_buf[path_slice], last_val)#路径与当前路径的价值进行链接
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        # 计算时序差分
        # 通过取当前奖励rews[:-1]（不包括最后一个元素），加上折扣后的下一个价值self.gamma * vals[1:]，然后减去当前价值vals[:-1]来得到的
        # 这给出了每一步相对于当前价值估计的奖励偏差
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        #使用discount_cumsum函数计算折扣累积和（计算优势函数的值
        # 使用deltas作为输入，并应用折扣因子self.gamma * self.lam（结合了折扣因子和GAE-Lambda的λ）
        # 结果存储在adv_buf中当前路径的对应位置
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]        
        #用discount_cumsum函数计算奖励的折扣累积和（即奖励到目标）
        # 通过将奖励数组rews与折扣因子self.gamma一起传递给discount_cumsum来完成的
        #由于我们不需要最后一个时间步的折扣累积和（因为通常没有后续步骤来折扣），所以使用[:-1]来排除它。结果存储在ret_buf中当前路径的对应位置。
        self.path_start_idx = self.ptr
        #更新路径起始索引为self.ptr。这表示下一个存储的交互将开始一个新的路径，因此当前路径的所有数据都已被处理

    def get(self):     
        assert self.ptr == self.max_size    # buffer has to be full before you can get确保经验池是满的

        self.ptr, self.path_start_idx = 0, 0#重置指针和路径索引（相当于清空经验池

        act_ten = torch.as_tensor(self.act_buf, dtype=torch.float32)
        ret_ten = torch.as_tensor(self.ret_buf, dtype=torch.float32)
        adv_ten = torch.as_tensor(self.adv_buf, dtype=torch.float32)
        logp_ten = torch.as_tensor(self.logp_buf, dtype=torch.float32)
        #将缓冲区中的动作（actions）、回报（returns）、优势（advantages）和对数概率（log probabilities）转换为PyTorch张量，并将数据类型设置为torch.float32
        obs_tensor_list = list(map(lambda o: torch.as_tensor(o, dtype=torch.float32), self.obs_buf))
        #将观测值处理为张量

        data = dict(obs=obs_tensor_list, act=act_ten, ret=ret_ten,
                    adv=adv_ten, logp=logp_ten)
        #所有数据（观测值、动作、回报、优势和对数概率）都被组合成一个字典，以便于后续的访问和使用

        return data

    def complete(self):
        self.ptr, self.path_start_idx = 0, 0#重置缓冲区中的两个指针

class multi_ppo:
    def __init__(self, env, ac_policy, pi_lr=3e-4, vf_lr=1e-3, train_epoch=50, steps_per_epoch = 600, max_ep_len=300, gamma=0.99, lam=0.97, clip_ratio=0.2, train_pi_iters=100, train_v_iters=100, target_kl=0.01, render=False, render_freq=20, con_train=False, seed=7, save_freq=50, save_figure=False, save_path='test/', save_name='test', load_fname=None, use_gpu = False, reset_mode=1, save_result=False, counter=0, test_env=None, lr_decay_epoch=1000, max_update_num=10, mpi=False, figure_save_path=None, **kwargs):

        torch.manual_seed(seed)#分别设置了PyTorch的CPU随机数生成器、PyTorch的GPU随机数生成器和NumPy的随机数生成器的种子
        torch.cuda.manual_seed(seed) 
        np.random.seed(seed)

        self.env = env#环境
        self.ac = ac_policy#策略网络
        self.con_train=con_train#是否继续训练
        self.robot_num = env.ir_gym.robot_number#机器人数量
        self.reset_mode = reset_mode#是否重置环境

        self.obs_dim = env.observation_space.shape#观察和动作的维度
        self.act_dim = env.action_space.shape

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)#创建优化器（pi策略网络，v值函数）
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

        if con_train:
            check_point = torch.load(load_fname)#加载模型检查点
            self.ac.load_state_dict(check_point['model_state'], strict=True)#将模型状态加载到策略网络中
            self.ac.train()#训练
            # self.ac.eval()

        # parameter
        self.epoch = train_epoch#训练周期数
        self.max_ep_len = max_ep_len#最大步数
        self.steps_per_epoch = steps_per_epoch#经验缓冲区大小
        
        self.buf_list = [multi_PPObuf(self.obs_dim, self.act_dim, steps_per_epoch, gamma, lam) for i in range(self.robot_num)]#经验缓冲区列表
        #obs_dim, act_dim, size, gamma=0.99, lam=0.95)

        # update parameters
        self.clip_ratio = clip_ratio#剪裁比例
        self.train_pi_iters = train_pi_iters
        self.train_v_iters=train_v_iters
        self.target_kl=target_kl    

        self.render = render
        self.render_freq = render_freq

        self.save_freq = save_freq #保持频率 
        self.save_path = save_path#保持路径
        self.figure_save_path = figure_save_path
        self.save_name = save_name
        self.save_figure = save_figure  
        self.use_gpu = use_gpu 

        self.save_result = save_result
        self.counter = counter
        self.pt = post_train(test_env, reset_mode=reset_mode, inf_print=False, render=False)
        #这行代码创建了一个post_train对象的实例
        # post_train可能是一个用于在训练后执行某些操作（如评估模型、收集统计数据等）的类。
        # 它接收一个测试环境test_env和几个其他参数（如reset_mode、inf_print和render），这些参数定义了后训练处理的行为。
        torch.cuda.synchronize()#确保所有之前的CUDA（GPU）操作都已完成

        self.lr_decay_epoch = lr_decay_epoch
        self.max_update_num = max_update_num#学习率衰减的周期数（lr_decay_epoch）和最大更新次数（max_update_num

        self.mpi = mpi

        if self.mpi:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()#如果self.mpi为True，则代码会创建两个额外的实例变量：self.comm和self.ra

    def training_loop(self):

        obs_list, ep_ret_list, ep_len_list = self.env.reset(mode=self.reset_mode), [0] * self.robot_num, [0] * self.robot_num
        #obs_list`: 从环境中重置并获取初始观察值列表
        # ep_ret_list, ep_len_list初始化两个列表来跟踪每个机器人的每个回合的回报和长度，初始值均为0
        ep_ret_list_mean = [[] for i in range(self.robot_num)]#每个机器人的平均回合回报

        for epoch in range(self.epoch + 1):
            start_time = time.time()
            print('current epoch', epoch)

            if self.mpi:
                state_dict = self.comm.bcast(self.ac.state_dict(), root=0)
                self.ac.load_state_dict(state_dict)#如果使用了MPI（多进程接口），则从根进程（通常是进程0）广播模型参数，并加载到当前进程的模型中。

            for t in range(self.steps_per_epoch):#

                if self.render and (epoch % self.render_freq == 0 or epoch == self.epoch):
                    self.env.render(save=self.save_figure, path=self.figure_save_path, i = t )#渲染环境

                # if self.save_figure and epoch == 1:
                #     self.env.render(save=True, path=self.save_path+'figure/', i=t)

                a_list, v_list, logp_list, abs_action_list = [], [], [], []
            
                for i in range(self.robot_num):#对于每个机器人，代码从环境中获取观察值，并使用策略（self.ac）计算动作、值和对数概率。它还计算绝对动作（考虑了当前速度）。
                    obs = obs_list[i]#获取观察值

                    a_inc, v, logp = self.ac.step(torch.as_tensor(obs, dtype=torch.float32))
                    a_inc = np.round(a_inc, 2)#从策略（self.ac）中得到的动作增量
                    a_list.append(a_inc)
                    v_list.append(v)#从策略中得到的值估计
                    logp_list.append(logp)#动作的对数概率（log probability）。它表示在给定状态下选择该动作的概率的对数值。

                    cur_vel = np.squeeze(self.env.ir_gym.robot_list[i].vel_omni)
                    abs_action = self.env.ir_gym.acceler * np.round(a_inc, 2)  + cur_vel#动作=动作增量+当前动作
                    # abs_action = 1.5*a_inc
                    abs_action = np.round(abs_action, 2)
                    abs_action_list.append(abs_action)

                next_obs_list, reward_list, done_list, info_list = self.env.step_ir(abs_action_list, vel_type = 'omni')#并获取新的观察值、奖励、结束标志和其他信息

                # save to buffer
                for i in range(self.robot_num):
                    
                    self.buf_list[i].store(obs_list[i], a_list[i], reward_list[i], v_list[i], logp_list[i])#将当前的观察值、动作、奖励、值和动作的对数概率存储到对应的经验回放缓冲区
                    ep_ret_list[i] += reward_list[i]#更新回合汇报
                    ep_len_list[i] += 1#更新回合计数

                # Update obs 
                obs_list = next_obs_list[:]#获取下一观测列表

                epoch_ended = t == self.steps_per_epoch-1#：当前迭代是epoch的最后一个迭代。
                arrive_all = min(info_list) == True#所有无人机到达
                terminal = max(done_list) == True or max(ep_len_list) > self.max_ep_len#至少有一个机器人达到了终止条件（done_list[i] 为 True）或任何一个机器人的回合长度超过了最大回合长度

                if epoch_ended or arrive_all:#如果回合结束，或无人机全部到达：重置环境、清空回合回报和长度、完成路径

                    if epoch + 1 % 300 == 0:
                        obs_list = self.env.reset(mode=self.reset_mode)
                    else:
                        obs_list = self.env.reset(mode=0)
                    
                    for i in range(self.robot_num):
                        
                        if arrive_all:
                            ep_ret_list_mean[i].append(ep_ret_list[i])

                        ep_ret_list[i] = 0
                        ep_len_list[i] = 0

                        self.buf_list[i].finish_path(0)#结束路径并切片计算回报

                elif terminal:#如果某个机器人到达结束条件：重置单个机器人、保存回合回报、清空回合回报和长度、完成路径

                    for i in range(self.robot_num):
                        if done_list[i] or ep_len_list[i] > self.max_ep_len:
                        
                            self.env.reset_one(i)
                            ep_ret_list_mean[i].append(ep_ret_list[i])
                            ep_ret_list[i] = 0
                            ep_len_list[i]= 0

                        self.buf_list[i].finish_path(0)
                    
                    obs_list = self.env.ir_gym.env_observation()#某个机器人达到了终止条件，但不是所有的机器人都完成了，代码将获取所有机器人当前的新观察值

            if (epoch % self.save_freq == 0) or (epoch == self.epoch):
                self.save_model(epoch) #在每个指定的epoch间隔（self.save_freq）或达到最大epoch时，代码将保存当前模型的参数

                if self.save_result and epoch != 0:#如果self.save_result为真且不是第一个epoch（epoch != 0），则会启动一个新线程来测试保存的模型，并将测试结果保存在指定的文件中
                # if self.save_result:
                    policy_model = self.save_path + self.save_name+'_'+str(epoch)+'.pt'#### ... 保存模型文件名设置 ...  
                    # policy_model = self.save_path + self.save_name+'_'+'check_point_'+ str(epoch)+'.pt'
                    result_path = self.save_path
                    policy_name = self.save_name+'_'+str(epoch)
                    thread = threading.Thread(target=self.pt.policy_test, args=('drl', policy_model, policy_name, result_path, '/results.txt'))
                    thread.start()

            mean = [round(np.mean(r), 2) for r in ep_ret_list_mean]               
            max_ret = [round(np.max(r), 2) for r in ep_ret_list_mean]   
            min_ret = [round(np.min(r), 2) for r in ep_ret_list_mean]   
            print('The reward in this epoch: ', 'min', min_ret, 'mean', mean, 'max', max_ret)#回合奖励的均值、最大值和最小值
            ep_ret_list_mean = [[] for i in range(self.robot_num)]#重置用于存储每个机器人回合奖励的列表

            # update
            # self.update()
            data_list = [buf.get() for buf in self.buf_list]#从每个机器人的经验回放缓冲区中收集数据
            if self.mpi:#如果使用mpi
                rank_data_list = self.comm.gather(data_list, root=0)#从根节点上收集所有进程的数据

                if self.rank == 0:
                    for data_list in rank_data_list:
                        self.update(data_list)
            else:
                self.update(data_list)#否则直接在当前进程中更新模型
    
            # animate
            # if epoch == 1:
            #     self.env.create_animate(self.save_path+'figure/')
            #计算了一个epoch的时间成本，并打印出来。同时，它还估算了剩余时间（以小时为单位）。注意，在MPI环境中，只有在根节点（self.rank == 0）上才会打印这些信息
            if self.mpi:
                if self.rank == 0:
                    time_cost = time.time()-start_time 
                    print('time cost in one epoch', time_cost, 'estimated remain time', time_cost*(self.epoch-epoch)/3600, 'hours' )
            else:
                time_cost = time.time()-start_time 
                print('time cost in one epoch', time_cost, 'estimated remain time', time_cost*(self.epoch-epoch)/3600, 'hours' )
            
    def update(self, data_list):
        
        randn = np.arange(self.robot_num)#0到self.robot_num-1的随机数组
        np.random.shuffle(randn)#打乱顺序（减少数据顺序偏见
        
        update_num = 0
        for r in randn:  
            
            data = data_list[r]##获取数据
            update_num += 1

            if update_num > self.max_update_num:
                continue

            for i in range(self.train_pi_iters):
                self.pi_optimizer.zero_grad()#清除之前的梯度
                loss_pi, pi_info = self.compute_loss_pi(data)#计算策略网络的损失
                #受一个包含观测值（obs）、动作（act）、优势函数（adv）和旧的对数概率（logp_old）的数据字典
                kl = pi_info['kl']
               
                
                if kl > self.target_kl:
                    print('Early stopping at step %d due to reaching max kl.'%i)
                    break
                
                loss_pi.backward()#反向传播
                self.pi_optimizer.step()#更新策略网络的参数

            # Value function learning
            for i in range(self.train_v_iters):
                self.vf_optimizer.zero_grad()
                loss_v = self.compute_loss_v(data)#计算价值网络的损失
                #接受一个包含观测值（obs）和回报（ret）的数据字典，并返回价值网络的预测值与真实回报之间的均方误差。如果使用了GPU，则将回报张量移动到GPU上
                loss_v.backward()
                self.vf_optimizer.step()


    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']#观测值，回报
        if self.use_gpu:
            ret = ret.cuda()
        return ((self.ac.v(obs) - ret)**2).mean()#计算价值网络（self.ac.v(obs)）的预测值与真实回报之间的均方误差，并返回该误差的均值

    def compute_loss_pi(self, data):#计算策略损失
         # Set up function for computing PPO policy loss
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']#含观测值（obs）、动作（act）、优势函数（adv）和旧的对数概率（logp_old

        if self.use_gpu:
            logp_old = logp_old.cuda()
            adv = adv.cuda()

        # Policy loss
        pi, logp = self.ac.pi(obs, act)#通过self.ac.pi(obs, act)得到新策略下的动作概率（pi）和对数概率（logp）
        ratio = torch.exp(logp - logp_old)#计算新旧策略的对数概率之差，并通过torch.exp得到比率（ratio
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv#通过裁剪比率（clip_adv），使得在优势函数为正的时候不会因比率过大而导致过大的更新
        #同样在优势函数为负的时候也不会因比率过小而过度惩罚
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()#策略损失（loss_pi）是通过取未裁剪比率乘以优势函数和裁剪优势函数中的较小值，并取均值得到的。

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        #算近似KL散度（approx_kl）、策略的熵（ent）以及裁剪的比例（clipfrac），并将这些信息存储在字典pi_info中返回
        return loss_pi, pi_info

    def save_model(self, index=0):
       
        dir_name = self.save_path
        fname_model = self.save_path + self.save_name+'_{}.pt'
        fname_check_point = self.save_path + self.save_name+'_check_point_{}.pt'
        state_dict = dict(model_state=self.ac.state_dict(), pi_optimizer=self.pi_optimizer.state_dict(), 
        vf_optimizer = self.vf_optimizer.state_dict() )

        if os.path.exists(dir_name):
            torch.save(self.ac, fname_model.format(index))
            torch.save(state_dict, fname_check_point.format(index))
        else:
            os.makedirs(dir_name)
            torch.save(self.ac, fname_model.format(index))
            torch.save(state_dict, fname_check_point.format(index))
                    

                
                
                  

