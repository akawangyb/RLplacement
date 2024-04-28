# --------------------------------------------------
# 文件名: test_maddpg
# 创建时间: 2024/4/23 9:21
# 描述: 测试训练好的maddpg模型的性能
# 作者: WangYuanbo
# --------------------------------------------------
import os
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from customENV.env_v1 import CustomEnv
from tools import gumbel_softmax, onehot_from_logits, to_maxtrix, to_list


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

        # return action.detach().cpu().numpy()[0]
        return action.detach()

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)

    def save(self, name):
        torch.save(self.actor, name)


class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau, log_dir):
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
        self.log_dir = log_dir

    @property
    def actors(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_actors(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, state, explore):
        # 所有的智能体公用同一个系统状态
        # 这里假设输入state已经是一个tensor了
        actions = [
            agent.take_action(state, explore) for agent in self.agents
        ]
        return actions

    def update(self, sample, i_agent, weights=None):
        # sample的第一个维度是智能体
        state, placing_actions, routing_actions, rewards, next_state, done = sample
        batch_size = state.shape[0]
        # 假设这里的采样已经是i_agent的transition
        cur_agent = self.agents[i_agent]
        # 更新每个智能体的评价网络
        # 更新规则，先用目标网络（策略和评价）计算下一个状态的价值
        # 用reward + gamma 表示下一个状态的未来价值A
        # 损失函数
        # 用评价网络计算当前（状态，动作）的价值，计算其与A之间的损失

        # 计算损失之前先把梯度清零
        cur_agent.critic_optimizer.zero_grad()
        # 计算当前智能体的cur_agent.target_actor(next_states)
        # 计算target_policy的输入
        all_next_actions = [
            onehot_from_logits(agent_actor(next_state))
            for agent_actor in self.target_actors
        ]
        target_critic_input = torch.cat((next_state, *all_next_actions), dim=1)

        immediate_return = cur_agent.target_critic(target_critic_input)

        # 这里的第一维是buffer_size
        delayed_return = rewards[:, i_agent].view(-1, 1) + self.gamma * immediate_return * (1 - done.view(-1, 1))
        placing_actions = placing_actions.view(batch_size, -1)
        routing_actions = routing_actions.view(batch_size, -1)
        critic_input = torch.cat((state, placing_actions, routing_actions), dim=1)
        critic_value = cur_agent.critic(critic_input)
        if weights is None:
            weights = torch.ones_like(critic_value)
        # 计算td_error
        td_error = torch.abs(critic_value - delayed_return).detach()
        critic_loss = torch.mean((critic_value - delayed_return.detach()) ** 2 * weights)
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        # 更新动作网络
        # 先计算全体智能体的动作网络生成的动作
        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(state)
        cur_act_vf_in = gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, agent_actor in enumerate(self.actors):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(agent_actor(state)))

        vf_in = torch.cat((state, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

        return critic_loss.item(), td_error

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)

    def save(self):
        for id, agent in enumerate(self.agents):
            name = 'agent' + str(id) + '.pkl'
            name = os.path.join(self.log_dir, name)
            agent.save(name)

    def load_model(self):
        for id, agent in enumerate(self.agents):
            name = 'agent' + str(id) + '.pkl'
            name = os.path.join(self.log_dir, name)
            agent.actor = torch.load(name)


Config = namedtuple('Config',
                    ['num_episodes',
                     'target_update',
                     'buffer_size',
                     'minimal_size',
                     'batch_size',
                     'actor_lr',
                     'critic_lr',
                     'update_interval',
                     'hidden_dim',
                     'gamma',
                     'tau'
                     ])

with open('train_config.yaml', 'r', encoding='utf-8') as f:
    config_data = yaml.safe_load(f)

config = Config(**config_data)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
env = CustomEnv()
routing_action_dims = env.routing_action_dims
placing_action_dims = env.placing_action_dims
action_dims = placing_action_dims + routing_action_dims
state_dims = env.state_dims

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
log_dir = 'log_reserved/pmr_maddpg_env-v1-20240422-223735'
critic_input_dim = env.state_dim + sum(action_dims)
maddpg = MADDPG(env=env, action_dims=action_dims, state_dims=state_dims,
                critic_input_dim=critic_input_dim,
                hidden_dim=config.hidden_dim, actor_lr=config.actor_lr, critic_lr=config.critic_lr,
                tau=config.tau, gamma=config.gamma, device=device, log_dir=log_dir)

maddpg.load_model()
# 开始测试
state, done = env.reset()
returns = np.zeros(env.agents_number)
while not done:
    state_Tensor = env.state_to_tensor().view(1, -1)
    actions = maddpg.take_action(state_Tensor, explore=True)
    placing_actions = actions[:env.placing_agents_number]
    routing_actions = actions[-env.routing_agents_number:]
    placing_matrix = to_maxtrix(placing_actions, (env.server_number, env.container_number))  # 要把这个tensor转换成一个矩阵
    # print("placing_matrix:", placing_matrix)
    routing_list = to_list(routing_actions)
    env_action = {
        'placing_action': placing_matrix,
        'routing_action': routing_list,
    }

    # 从actions里面获得env可以识别的action
    # 这里输出的actions是一个list的np array
    # 主要是这一步，action与环境进行交互
    print("placing_action:", placing_matrix)
    print("routing_action:", routing_list)

    next_state, rewards, done, info = env.step(env_action)
    returns += info
    print(info.tolist())
print(returns.tolist())
