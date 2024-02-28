# --------------------------------------------------
# 文件名: routing_agent
# 创建时间: 2024/2/26 10:20
# 描述: 用户请求的智能体
# 作者: WangYuanbo
# --------------------------------------------------
# 一个用户请求要路由到s个服务器上，动作空间是s+1
# n个用户的请求要路由，动作空间是(s+1)^n
# 这个问题的奖励不是稀疏的
# 假设一个解决用户的请求路由到哪个服务器上，
import argparse
import collections
import os
import random
import sys
from collections import namedtuple, deque
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from routing_env import CustomEnv
from tools import get_server_info, get_container_info, get_user_request_info, get_placing_info

# 经验是一个具名元组
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# 首先定义经验回放池
class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        state = np.array(state)
        return state, action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class ReplayMemory(object):
    # memory实现了一个队列
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        return self.memory[item]


def onehot_from_logits(logits, eps=0.2):
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(
        torch.eye(logits.shape[1])[[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]],
        requires_grad=False).to(logits.device)

    # for i, r in enumerate(torch.rand(logits.shape[0])):
    #     print(i, r)
    # print(argmax_acs.shape, rand_acs.shape)
    # print(argmax_acs.shape[0], argmax_acs.shape[1])
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# 评估动作，状态的价值
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    ''' DDPG算法 '''

    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = TwoLayerFC(state_dim, action_dim,
                                       hidden_dim).to(device)

        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1,
                                        hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

    def take_action(self, state, explore=False):

        action = self.actor(state)

        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)


class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau):
        self.agents = []
        for i in range(env.agents):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        self.container_number = env.container_number

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def dict_to_np(self, states):
        a = states['server_cpu']
        b = states['user_request_cpu']
        c = np.eye(self.container_number)[states['user_request_imageID']].reshape(-1)
        states = np.concatenate((a, b, c))
        states = states.reshape(1, -1)
        # states = torch.tensor(states, dtype=torch.float).view(1, -1).to(self.device)
        return states

    def take_action(self, states, explore):
        # 所有的子智能体公用同一个系统状态
        # 系统状态是Dict类型，直接铺平
        states = [self.dict_to_np(state) for state in states]

        states = [torch.tensor(state, dtype=torch.float, device=self.device)
                  for state in states]
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent):
        # sample的第一个维度是智能体
        states, actions, rewards, next_states, dones = sample
        # 这些数据还是list类型没有换成tensor
        # batch_size = len(states)
        # 第一维是智能体数量,第二位是取样批次大小
        # print("batch_size: ", batch_size)
        temp_states = []
        for agent_state in states:
            agent_state_tensor = []
            for state in agent_state:
                agent_state_tensor.append(torch.tensor(self.dict_to_np(state), dtype=torch.float).to(self.device))
            temp_states.append(torch.cat(agent_state_tensor, dim=0))
        states = temp_states

        # actions = torch.stack([torch.tensor(action, dtype=torch.float).to(self.device) for action in actions])
        actions = torch.tensor(np.array(actions), dtype=torch.float).to(self.device)

        # print(actions.shape)
        rewards = torch.stack([torch.tensor(reward, dtype=torch.float).to(self.device) for reward in rewards])

        temp_states = []
        for agent_state in next_states:
            agent_state_tensor = []
            for state in agent_state:
                agent_state_tensor.append(torch.tensor(self.dict_to_np(state), dtype=torch.float).to(self.device))
            temp_states.append(torch.cat(agent_state_tensor, dim=0))
        next_states = temp_states

        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)

        cur_agent = self.agents[i_agent]
        # 更新每个智能体的评价网络
        # 更新规则，先用目标网络（策略和评价）计算下一个状态的价值
        # 用reward + gamma 表示下一个状态的未来价值A
        # 损失函数
        # 用评价网络计算当前（状态，动作）的价值，计算其与A之间的损失

        # 计算损失之前先把梯度清零
        cur_agent.critic_optimizer.zero_grad()
        # 计算所有智能体的cur_agent.target_actor(next_states)
        # next_states 是一个状态张量torch.Size([64, 1, 113])
        # print(type(next_states))
        all_target_actions = [
            onehot_from_logits(agent_critic(next_st))
            for agent_critic, next_st in zip(self.target_policies, next_states)
        ]

        target_critic_input = torch.cat((*next_states, *all_target_actions), dim=1)

        answer = cur_agent.target_critic(target_critic_input)
        target_critic_value = rewards[i_agent].view(-1, 1) + self.gamma * answer * (1 - dones[i_agent].view(-1, 1))

        critic_input = torch.cat((*states, *actions), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        # 更新策略网络
        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(states[i_agent])
        cur_act_vf_in = gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, states)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)))
        vf_in = torch.cat((*states, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)


# 指定训练gpu
parser = argparse.ArgumentParser(description='选择训练GPU的参数')
parser.add_argument('--gpu', type=int, default=0, help='要使用的GPU的编号')

# 解析参数
args = parser.parse_args()

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# 确认使用的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 获得脚本名字
filename = os.path.basename(sys.argv[0])  # 获取脚本文件名
script_name = os.path.splitext(filename)[0]  # 去除.py后缀
# 日志输出路径
father_log_directory = '../log'
if not os.path.exists(father_log_directory):
    os.makedirs(father_log_directory)
current_time = datetime.now()
formatted_time = current_time.strftime('%Y%m%d-%H_%M_%S')
log_file_name = script_name + formatted_time

log_path = os.path.join(father_log_directory, log_file_name)
# 规范化文件路径
log_dir = os.path.normpath(log_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'info.log')
f_log = open(log_path, 'w', encoding='utf-8')
writer = SummaryWriter(log_dir)

num_episodes = 10000
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
actor_lr = 3e-4
critic_lr = 3e-3
update_interval = 100
hidden_dim = 64
gamma = 0.98
tau = 0.005  # 软更新参数
sigma = 0.01  # 高斯噪声标准差

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
penalty = -10
server_info = get_server_info()
container_info = get_container_info()
user_request_info = get_user_request_info(config['user_number'])
placing_info = get_placing_info()

config['max_cpu'] = 5
env = CustomEnv(server_info=server_info, container_info=container_info, user_request_info=user_request_info,
                placing_info=placing_info, penalty=penalty, config=config)
raw_state = env.reset()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dims = env.state_dims
action_dims = env.action_dims
critic_input_dim = sum(state_dims) + sum(action_dims)

maddpg = MADDPG(env=env, state_dims=state_dims, action_dims=action_dims,
                critic_input_dim=critic_input_dim,
                hidden_dim=hidden_dim, actor_lr=actor_lr, critic_lr=critic_lr,
                tau=tau, gamma=gamma, device=device)


def evaluate(para_env, maddpg, n_episode=10):
    # 对学习的策略进行评估,此时不会进行探索
    env = para_env
    returns = np.zeros(env.agents)
    for _ in range(n_episode):
        states, dones = env.reset()
        while not dones[0]:
            actions = maddpg.take_action(states, explore=False)
            next_states, rewards, dones, info = env.step(actions)
            # print(rewards)
            rewards = np.array(rewards)
            returns += rewards * 1.0
    returns = returns / n_episode
    return returns.tolist()


return_list = []  # 记录每一轮的回报（return）
total_step = 0
for i_episode in range(num_episodes):
    state, done = env.reset()
    # ep_returns = np.zeros(len(env.agents))
    while not done[0]:
        actions = maddpg.take_action(state, explore=True)
        # 这里输出的actions是一个np的list
        next_state, reward, done, _ = env.step(actions)
        # print(reward)
        replay_buffer.add(state, actions, reward, next_state, done)
        state = next_state
        total_step += 1
        if replay_buffer.size(
        ) >= minimal_size and total_step % update_interval == 0:
            sample = replay_buffer.sample(batch_size)


            # 原始记忆的形状为（时间步长，智能体数，状态 / 行动 / 奖励）
            # 按智能体训练的记忆形状为 （智能体数，时间步长，状态/行动/奖励）
            # 方便获得每个智能体的状态/行动/奖励的序列

            # 这里记忆是按照时间排序的，可以理解为列的维度是时间，
            # 但是对于智能体的训练，需要分别单独更新
            # 现在要按智能体排序
            def stack_array(x):

                rearranged = [[sub_x[i] for sub_x in x]
                              for i in range(len(x[0]))]
                return rearranged


            sample = [stack_array(x) for x in sample]

            for a_i in range(env.agents):
                maddpg.update(sample, a_i)
            maddpg.update_all_targets()
    if (i_episode + 1) % 100 == 0:
        ep_returns = evaluate(env, maddpg, n_episode=100)
        f_log.write("Episode: {}, total reward: {}\n".format(i_episode + 1, sum(ep_returns)))
        f_log.write("each agent reward: {}\n".format(ep_returns))
        return_list.append(ep_returns)
        print(f"Episode: {i_episode + 1}, {ep_returns}")
