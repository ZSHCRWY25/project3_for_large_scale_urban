'''
:@Author: 刘玉璞
:@Date: 2024/6/26 16:45:20
:@LastEditors: 刘玉璞
:@LastEditTime: 2024/7/4 17:51:11
:Description: 
'''
import os
import sys
import gym
import pickle
import shutil
import argparse
from torch import nn
from pathlib import Path
from rl_rvo_nav.policy_train.multi_ppo import multi_ppo
from rl_rvo_nav.policy.policy_rnn_ac import rnn_ac

# path set
cur_path = Path(__file__).parent
world_abs_path = str(cur_path/'train_world.yaml')#指定世界配置文件的路径，默认为'train_world.yaml'

# default

counter = 0

# model_path = cur_path / 'model_save' / ('r' + str(robot_number) )

parser = argparse.ArgumentParser(description='drl rvo parameters')

par_env = parser.add_argument_group('par env', 'environment parameters') 
par_env.add_argument('--env_name', default='mrnav-v1')#指定环境名称
par_env.add_argument('--world_path', default='train_world.yaml')#指定世界配置文件的路径，默认为'train_world.yaml'
par_env.add_argument('--drone_number', type=int, default=4)#无人机数量
par_env.add_argument('--init_mode', default=3)#初始化模式
par_env.add_argument('--reset_mode', default=3)#重置模式
par_env.add_argument('--mpi', default=False)#: MPI（Message Passing Interface）启用标志

par_env.add_argument('--neighbors_region', default=4)#邻居区域
par_env.add_argument('--neighbors_num', type=int, default=5)   #邻居数量
par_env.add_argument('--reward_parameter', type=float, default=(3.0, 0.3, 0.0, 6.0, 0.3, 3.0, -0, 0), nargs='+')#奖励函数
par_env.add_argument('--env_train', default=True)#训练标志
par_env.add_argument('--random_bear', default=True)#随机方向
par_env.add_argument('--random_radius', default=False)#随机半径
par_env.add_argument('--full', default=False)

par_policy = parser.add_argument_group('par policy', 'policy parameters') 
par_policy.add_argument('--state_dim', default=6)#状态向量维度
par_policy.add_argument('--rnn_input_dim', default=8)#RNN输入维度
par_policy.add_argument('--rnn_hidden_dim', default=256)
par_policy.add_argument('--trans_input_dim', default=8)#输入维度
par_policy.add_argument('--trans_max_num', default=10)#最大序列
par_policy.add_argument('--trans_nhead', default=1)#多头注意力头数
par_policy.add_argument('--trans_mode', default='attn')#转换器模式
par_policy.add_argument('--hidden_sizes_ac', default=(256, 256))
par_policy.add_argument('--drop_p', type=float, default=0)#丢弃率
par_policy.add_argument('--hidden_sizes_v', type=tuple, default=(256, 256))  # 16 16
par_policy.add_argument('--activation', default=nn.ReLU)
par_policy.add_argument('--output_activation', default=nn.Tanh)
par_policy.add_argument('--output_activation_v', default=nn.Identity)
par_policy.add_argument('--use_gpu', action='store_true')   
par_policy.add_argument('--rnn_mode', default='biGRU')   # LSTM

par_train = parser.add_argument_group('par train', 'train parameters') 
par_train.add_argument('--pi_lr', type=float, default=4e-6)#policy学习率
par_train.add_argument('--vf_lr', type=float, default=5e-5)#value学习率
par_train.add_argument('--train_epoch', type=int, default=250)
par_train.add_argument('--steps_per_epoch', type=int, default=500)
par_train.add_argument('--max_ep_len', default=150)#最大执行长度
par_train.add_argument('--gamma', default=0.99)#折扣因子
par_train.add_argument('--lam', default=0.97)#GAE参数
par_train.add_argument('--clip_ratio', default=0.2)#梯度裁剪比例
par_train.add_argument('--train_pi_iters', default=50)
par_train.add_argument('--train_v_iters', default=50)#训练迭代次数
par_train.add_argument('--target_kl',type=float, default=0.05)
par_train.add_argument('--render', default=True)
par_train.add_argument('--render_freq', default=50)#渲染频率
par_train.add_argument('--con_train', action='store_true')#连续训练
par_train.add_argument('--seed', default=7)#随机种子（用于生成环境
par_train.add_argument('--save_freq', default=50)
par_train.add_argument('--save_figure', default=False)
par_train.add_argument('--figure_save_path', default='figure')
par_train.add_argument('--save_path', default=str(cur_path / 'model_save') + '/')
par_train.add_argument('--save_name', default= 'r')
par_train.add_argument('--load_path', default=str(cur_path / 'model_save')+ '/')
par_train.add_argument('--load_name', default='r4_0/r4_0_check_point_250.pt') # '/r4_0/r4_0_check_point_250.pt' 模型保存和加载的文件名
par_train.add_argument('--save_result', type=bool, default=True)
par_train.add_argument('--lr_decay_epoch', type=int, default=1000)
par_train.add_argument('--max_update_num', type=int, default=10)#学习率衰减的Epoch数和最大更新次数

args = parser.parse_args()

# decide the model path and model name 
model_path_check = args.save_path + args.save_name + str(args.drone_number) + '_{}'
model_name_check = args.save_name + str(args.drone_number) +  '_{}'
while os.path.isdir(model_path_check.format(counter)):
    counter+=1

model_abs_path = model_path_check.format(counter) + '/'
model_name = model_name_check.format(counter)

load_fname = args.load_path + args.load_name

env = gym.make(args.env_name, world_name=args.world_path, robot_number=args.drone_number, neighbors_region=args.neighbors_region, 
               neighbors_num=args.neighbors_num, robot_init_mode=args.init_mode, env_train=args.env_train, random_bear=args.random_bear, 
               random_radius=args.random_radius, reward_parameter=args.reward_parameter, full=args.full)

test_env = gym.make(args.env_name, world_name=args.world_path, robot_number=args.drone_number, neighbors_region=args.neighbors_region, 
                    neighbors_num=args.neighbors_num, robot_init_mode=args.init_mode, env_train=False, random_bear=args.random_bear, 
                    random_radius=args.random_radius, reward_parameter=args.reward_parameter, plot=False, full=args.full)

policy = rnn_ac(env.observation_space, env.action_space, args.state_dim, args.rnn_input_dim, args.rnn_hidden_dim, 
                    args.hidden_sizes_ac, args.hidden_sizes_v, args.activation, args.output_activation, 
                    args.output_activation_v, args.use_gpu, args.rnn_mode, args.drop_p)

ppo = multi_ppo(env, policy, args.pi_lr, args.vf_lr, args.train_epoch, args.steps_per_epoch, args.max_ep_len, args.gamma, 
                args.lam, args.clip_ratio, args.train_pi_iters, args.train_v_iters, args.target_kl, args.render, args.render_freq,
                  args.con_train,  args.seed, args.save_freq, args.save_figure, model_abs_path, model_name, load_fname, args.use_gpu, 
                  args.reset_mode, args.save_result, counter, test_env, args.lr_decay_epoch, args.max_update_num, args.mpi, args.figure_save_path)

# save hyparameters
if not os.path.exists(model_abs_path):
    os.makedirs(model_abs_path)

f = open(model_abs_path + model_name, 'wb')
pickle.dump(args, f)
f.close()

with open(model_abs_path+model_name+'.txt', 'w') as p:
    print(vars(args), file=p)
p.close()

shutil.copyfile( str(cur_path/'train_world.yaml'), model_abs_path+model_name+'_world.yaml')

# run the training loop
ppo.training_loop()

