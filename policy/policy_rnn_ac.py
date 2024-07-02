import torch
import torch.nn as nn
import numpy as np
from gym.spaces import Box
from torch.distributions.normal import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import time

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []

    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation#如果当前层不是倒数第二层，act 将使用 activation 函数；否则，它将使用 output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]

    return nn.Sequential(*layers)

def mlp3(sizes, activation=nn.ReLU, output_activation=nn.Identity, drop_p = 0.5):

    layers = []
    for j in range(len(sizes)-1):
        if j < len(sizes)-3:
            act = activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), nn.Dropout(p=drop_p), act()]
        else:
            layers += [nn.Linear(sizes[j], sizes[j+1]), nn.Tanh()] 

    return nn.Sequential(*layers)

class rnn_ac(nn.Module):

    def __init__(self, observation_space, action_space, state_dim, rnn_input_dim=4, 
    rnn_hidden_dim=64, hidden_sizes_ac=(256, 256), hidden_sizes_v=(16, 16), 
    activation=nn.ReLU, output_activation=nn.Tanh, output_activation_v= nn.Identity, use_gpu=False, rnn_mode='GRU', drop_p=0):
        super().__init__()

        self.use_gpu = use_gpu
        torch.cuda.synchronize()#用于指示是否使用GPU。
        
        if rnn_mode == 'biGRU':#根据 rnn_mode 的不同，计算了 obs_dim，表示观测维度,根据 rnn_mode 的不同，将状态维度和RNN隐藏层维度相加，得到 obs_dim
            obs_dim = (rnn_hidden_dim + state_dim)
        elif rnn_mode == 'GRU' or 'LSTM':
            obs_dim = (rnn_hidden_dim + state_dim)

        rnn = rnn_Reader(state_dim, rnn_input_dim, rnn_hidden_dim, use_gpu=use_gpu, mode=rnn_mode)#创建了一个RNN模型（rnn_Reader），用于处理状态信息。

        # policy builder depends on action space
        if isinstance(action_space, Box):#动作空间是连续的（Box类型），则创建一个高斯策略网络（GaussianActor）。
            self.pi = GaussianActor(obs_dim, action_space.shape[0], hidden_sizes_ac, activation, output_activation, rnn_reader=rnn, use_gpu=use_gpu)

        # build value function
        self.v = Critic(obs_dim, hidden_sizes_v, activation, output_activation_v, rnn_reader=rnn, use_gpu=use_gpu)#值函数网络（Critic


    def step(self, obs, std_factor=1):
        with torch.no_grad():#obs：表示环境观测的输入
            pi_dis = self.pi._distribution(obs, std_factor)#从策略网络（self.pi）中获取动作分布（pi_dis）
            a = pi_dis.sample()#从分布中采样一个动作（a
            logp_a = self.pi._log_prob_from_distribution(pi_dis, a)#计算该动作的对数概率（logp_a
            v = self.v(obs)#从值函数网络（self.v）中获取状态值（v

            if self.use_gpu:#如果启用了GPU，将结果转移到CPU上
                a = a.cpu()
                logp_a = logp_a.cpu()
                v = v.cpu()

        return a.numpy(), v.numpy(), logp_a.numpy()#返回动作、状态值和对数概率的NumPy数组。

    def act(self, obs, std_factor=1):#对 step 函数的封装，只返回动作
        return self.step(obs, std_factor)[0]


class rnn_Reader(nn.Module):#用于处理状态信息
    def __init__(self, state_dim, input_dim, hidden_dim, use_gpu=False, mode='GRU'):
        super().__init__()
        
        self.state_dim = state_dim
        self.mode = mode

        if mode == 'GRU':
            self.rnn_net = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif mode == 'LSTM':
            self.rnn_net = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        elif mode == 'biGRU':
            self.rnn_net = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.use_gpu=use_gpu

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        des_dim = state_dim + hidden_dim
        self.ln = nn.LayerNorm(des_dim)#self.ln 是一个LayerNorm层，用于对输入进行标准化。

        if use_gpu: 
            self.rnn_net = self.rnn_net.cuda()
            self.ln = self.ln.cuda()
            # self.conv = self.conv.cuda()

    def obs_rnn(self, obs):

        obs = torch.as_tensor(obs, dtype=torch.float32)  #将输入的 obs 转换为PyTorch张量，并指定数据类型为 float32

        if self.use_gpu:#如果启用了GPU，将张量移动到GPU上
            obs=obs.cuda() 

        moving_state = obs[self.state_dim:]#表示移动状态的部分
        robot_state = obs[:self.state_dim]#表示机器人状态
        mov_len = int(moving_state.size()[0] / self.input_dim)#计算 moving_state 的长度，并将其重新形状为 (1, mov_len, self.input_dim)，以便作为RNN的输入
        rnn_input = torch.reshape(moving_state, (1, mov_len, self.input_dim))

        if self.mode == 'GRU' :
            output, hn = self.rnn_net(rnn_input)
        elif self.mode == 'biGRU':
            output, hn = self.rnn_net(rnn_input)
        elif self.mode == 'LSTM':
            output, (hn, cn) = self.rnn_net(rnn_input)
    
        hnv = torch.squeeze(hn)#压缩隐藏状态 hn，以便在不同模式下使用
        if self.mode == 'biGRU':#如果是双向GRU模式，将隐藏状态求和
            hnv = torch.sum(hnv, 0)
        
        rnn_obs = torch.cat((robot_state, hnv))#将机器人状态 robot_state 和处理后的隐藏状态 hnv 连接起来
        rnn_obs = self.ln(rnn_obs)#使用LayerNorm层 self.ln 对连接后的状态进行标准化

        return rnn_obs  #返回经过处理的RNN状态表示 rnn_obs

    def obs_rnn_list(self, obs_tensor_list):
        
        mov_len = [(len(obs_tensor)-self.state_dim)/self.input_dim for obs_tensor in obs_tensor_list]#计算每个状态张量的移动状态长度，即去除机器人状态后的部分
        obs_pad = pad_sequence(obs_tensor_list, batch_first = True)#使用 pad_sequence 函数将状态张量列表 obs_tensor_list 进行填充，使其具有相同的长度。填充后的张量为 obs_pad
        robot_state_batch = obs_pad[:, :self.state_dim] #提取机器人状态部分，即前 self.state_dim 个元素
        batch_size = len(obs_tensor_list)
        if self.use_gpu:
            robot_state_batch=robot_state_batch.cuda()

        def obs_tensor_reform(obs_tensor):#定义了一个内部函数 obs_tensor_reform，用于将移动状态部分重新形状为 (mov_tensor_len, self.input_dim)
            mov_tensor = obs_tensor[self.state_dim:]
            mov_tensor_len = int(len(mov_tensor)/self.input_dim)
            re_mov_tensor = torch.reshape(mov_tensor, (mov_tensor_len, self.input_dim)) 
            return re_mov_tensor
        
        re_mov_list = list(map(lambda o: obs_tensor_reform(o), obs_tensor_list))#对每个状态张量应用 obs_tensor_reform 函数，得到一个新的状态列表 re_mov_list
        re_mov_pad = pad_sequence(re_mov_list, batch_first = True)#用 pad_sequence 函数对 re_mov_list 进行填充，得到填充后的张量 re_mov_pad。

        if self.use_gpu:
            re_mov_pad=re_mov_pad.cuda()

        moving_state_pack = pack_padded_sequence(re_mov_pad, mov_len, batch_first=True, enforce_sorted=False)
        
        if self.mode == 'GRU':#获取RNN的输出 output 和隐藏状态 hn（或隐藏状态和细胞状态 hn 和 cn）
            output, hn= self.rnn_net(moving_state_pack)
        elif self.mode == 'biGRU':
            output, hn= self.rnn_net(moving_state_pack)
        elif self.mode == 'LSTM':
            output, (hn, cn) = self.rnn_net(moving_state_pack)

        hnv = torch.squeeze(hn)#压缩隐藏状态 hn，以便在不同模式下使用。
#   如果是双向GRU模式，将隐藏状态求和。

        if self.mode == 'biGRU':
            hnv = torch.sum(hnv, 0)
        
        fc_obs_batch = torch.cat((robot_state_batch, hnv), 1)#将机器人状态 robot_state_batch 和处理后的隐藏状态 hnv 连接起来
        fc_obs_batch = self.ln(fc_obs_batch)#将机器人状态 robot_state_batch 和处理后的隐藏状态 hnv 连接起来

        return fc_obs_batch#返回经过处理的RNN状态表示 fc_obs_batch。

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None, std_factor=1):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs, std_factor)
        logp_a = None

        if act is not None:   
            logp_a = self._log_prob_from_distribution(pi, act)

        return pi, logp_a


class GaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation, rnn_reader=None, use_gpu=False):
        super().__init__()

        self.rnn_reader = rnn_reader
        self.use_gpu = use_gpu
        self.net_out=mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation)
        # self.net_out=mlp3([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation, 0)

        log_std = -1 * np.ones(act_dim, dtype=np.float32)

        if use_gpu:
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std, device=torch.device('cuda')))
            self.net_out=self.net_out.cuda()
        else:
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def _distribution(self, obs, std_factor=1):

        if isinstance(obs, list):
            obs = self.rnn_reader.obs_rnn_list(obs)
            net_out = self.net_out(obs)
        else:
            obs = self.rnn_reader.obs_rnn(obs)
            net_out = self.net_out(obs)
        
        mu = net_out 
        std = torch.exp(self.log_std)
        std = std_factor * std
        
        return Normal(mu, std)
        
    def _log_prob_from_distribution(self, pi, act):

        if self.use_gpu:
            act = act.cuda()

        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class Critic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation, output_activation, rnn_reader=None, use_gpu=False):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation, output_activation)

        if use_gpu:
            self.v_net = self.v_net.cuda()

        self.rnn_reader = rnn_reader

    def forward(self, obs):

        if self.rnn_reader != None:
            if isinstance(obs, list):
                obs = self.rnn_reader.obs_rnn_list(obs)
            else:
                obs = self.rnn_reader.obs_rnn(obs)
        v = torch.squeeze(self.v_net(obs), -1)

        return v 