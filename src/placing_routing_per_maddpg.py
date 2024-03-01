# --------------------------------------------------
# 文件名: placing_routing_per_maddpg
# 创建时间: 2024/3/1 22:25
# 描述: 基于优先记忆回放的maddpg解决服务部署和请求调度
# 作者: WangYuanbo
# --------------------------------------------------

# --------------------------------------------------
# 文件名: placing_routing_maddpg
# 创建时间: 2024/2/28 16:00
# 描述: 服务部署请求路由的maddpg算法
# 作者: WangYuanbo
# --------------------------------------------------
import argparse
import os
import sys
from datetime import datetime

import yaml
from torch.utils.tensorboard import SummaryWriter

from memory.buffer import PrioritizedReplayBuffer
from placing_routing_env import CustomEnv
from tools import *


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
        # 这里的MADDPG，相当于一个大的公司下面有2个部门（placing ，routing），每个部门下面有自己的员工
        self.agents = []
        for i in range(env.agents_number):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device))

        self.placing_agents_number = env.placing_agents_number
        self.routing_agents_number = env.routing_agents_number

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
        # 把所有的数据铺平
        a = states['server_cpu']
        b = states['user_request_cpu']
        # c = np.eye(self.container_number)[states['user_request_imageID']].reshape(-1)
        c = states['user_request_imageID']
        d = states['server_storage']
        e = states['container_storage']
        f = states['last_placing_action'].reshape(-1)

        states = np.concatenate((a, b, c, d, e, f))
        states = states.reshape(1, -1)
        # states = torch.tensor(states, dtype=torch.float).view(1, -1).to(self.device)
        return states

    def take_action(self, states, explore):
        # routing智能体公用同一个状态
        # placing智能体公用同一个状态
        # routing系统状态是Dict类型，只需要把他处理即可

        states = [self.dict_to_np(state) for state in states]

        states = [torch.tensor(state, dtype=torch.float, device=self.device).view(1, -1)
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
        temp_states = []
        for agent_state in states:
            agent_state_tensor = []
            for state in agent_state:
                agent_state_tensor.append(torch.tensor(self.dict_to_np(state), dtype=torch.float).to(self.device))
            temp_states.append(torch.cat(agent_state_tensor, dim=0))
        states = temp_states

        # actions = torch.stack([torch.tensor(action, dtype=torch.float).to(self.device) for action in actions])
        # for index, action in enumerate(actions):
        #     print(index, type(action), len(action), action)
        # actions是一个list，list类型
        placing_actions = np.array(actions[:self.placing_agents_number])
        routing_actions = np.array(actions[-self.routing_agents_number:])
        placing_actions = torch.tensor(placing_actions, dtype=torch.float).to(self.device)
        routing_actions = torch.tensor(routing_actions, dtype=torch.float).to(self.device)
        # actions = torch.tensor(actions, dtype=torch.float).to(self.device)

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
        # 计算当前智能体的cur_agent.target_actor(next_states)
        all_target_actions = [
            onehot_from_logits(agent_critic(next_st))
            for agent_critic, next_st in zip(self.target_policies, next_states)
        ]

        target_critic_input = torch.cat((*next_states, *all_target_actions), dim=1)

        answer = cur_agent.target_critic(target_critic_input)
        target_critic_value = rewards[i_agent].view(-1, 1) + self.gamma * answer * (1 - dones[i_agent].view(-1, 1))

        critic_input = torch.cat((*states, *placing_actions, *routing_actions), dim=1)
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

num_episodes = 20000
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

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
penalty = -1000
server_info = get_server_info()
container_info = get_container_info()
user_request_info = get_user_request_info(config['user_number'])
placing_info = get_placing_info()

config['max_cpu'] = 5
config['max_storage'] = 100000
config['cloud_delay'] = 10
config['edge_delay'] = 5

env = CustomEnv(server_info=server_info, container_info=container_info, user_request_info=user_request_info,
                penalty=penalty, config=config)
raw_state = env.reset()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# 有两个动作
replay_buffer = PrioritizedReplayBuffer(
    state_size=env.state_dim,
    action_size=2,
    buffer_size=buffer_size)

# 假设有两个maddpg大agent,一个是routing组长，一个是placing组长,下面的全是员工

routing_action_dims = env.routing_action_dims
placing_action_dims = env.placing_action_dims
action_dims = placing_action_dims + routing_action_dims
state_dims = env.state_dims
critic_input_dim = sum(state_dims) + sum(action_dims)

maddpg = MADDPG(env=env, action_dims=action_dims, state_dims=state_dims,
                critic_input_dim=critic_input_dim,
                hidden_dim=hidden_dim, actor_lr=actor_lr, critic_lr=critic_lr,
                tau=tau, gamma=gamma, device=device)


def evaluate(para_env, maddpg, n_episode=10):
    # 对学习的策略进行评估,此时不会进行探索
    env = para_env
    returns = np.zeros(env.agents_number)
    for _ in range(n_episode):
        states, dones = env.reset()
        while not dones[0]:
            actions = maddpg.take_action(states, explore=False)
            next_states, rewards, dones, info = env.step(actions)
            rewards = np.array(rewards)
            returns += rewards * 1.0
    returns = returns / n_episode
    return returns.tolist()


return_list = []  # 记录每一轮的回报（return）
total_step = 0
for i_episode in range(num_episodes):
    states, dones = env.reset()
    # ep_returns = np.zeros(len(env.agents))
    while not dones[0]:
        actions = maddpg.take_action(states, explore=True)
        # 这里输出的actions是一个np的list
        next_states, rewards, dones, _ = env.step(actions)
        # print(rewards)
        # states 和 next_states 转换成一维的
        replay_buffer.add((states, actions, rewards, next_states, dones))
        states = next_states
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

            for a_i in range(env.agents_number):
                maddpg.update(sample, a_i)
            maddpg.update_all_targets()
    if (i_episode + 1) % 100 == 0:
        ep_returns = evaluate(env, maddpg, n_episode=100)
        f_log.write("Episode: {}, total reward: {}\n".format(i_episode + 1, sum(ep_returns)))
        f_log.write("each agent reward: {}\n".format(ep_returns))
        return_list.append(ep_returns)
        print(f"Episode: {i_episode + 1}, {ep_returns}")
