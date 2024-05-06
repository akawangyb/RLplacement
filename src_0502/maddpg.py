# --------------------------------------------------
# 文件名: maddpg
# 创建时间: 2024/5/2 12:07
# 描述:  maddpg算法描述
# 作者: WangYuanbo
# --------------------------------------------------
import os

import torch
import torch.nn.functional as F

from tools import onehot_from_logits, gumbel_softmax


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


class PlacingNet(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class RoutingDDPG:
    '''
     DDPG算法
     '''

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

        return action.detach()

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)

    def save(self, name):
        torch.save(self.actor, name)


class PlacingDDPG:
    '''
     DDPG算法
     '''

    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = PlacingNet(state_dim, action_dim, hidden_dim).to(device)
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
        """
        重写此处
        :param state:
        :param explore:
        :return:
        """
        action = self.actor(state)
        # if explore:
        #     # 添加噪声,进行随机探索
        noise = torch.randn_like(action).mul_(0.1)
        action += noise
        action = torch.clamp(action, min=0, max=1)
        # # 如果不是探索直接利用action
        # action_without_grad = (action > 0.5).float()
        # action_with_grad = (action_without_grad - action).detach() + action
        # return action_with_grad
        return action

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

        self.placing_agents_number = env.placing_agents_number
        self.routing_agents_number = env.routing_agents_number

        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        self.container_number = env.container_number
        self.log_dir = log_dir

        for i in range(env.placing_agents_number):
            self.agents.append(
                PlacingDDPG(state_dims[i], action_dims[i], critic_input_dim,
                            hidden_dim, actor_lr, critic_lr, device))

        for i in range(env.routing_agents_number):
            idx = i + env.placing_agents_number
            self.agents.append(
                RoutingDDPG(state_dims[idx], action_dims[idx], critic_input_dim,
                            hidden_dim, actor_lr, critic_lr, device))

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

    def update(self, idx, batch):
        state, actions, rewards, next_state, done = batch
        batch_size = state.shape[0]
        '''
        step1 对于每个智能体中心化训练全局critic网络
        step2 对于训练自身的actor网络
        step3 更新目标actor
        step4 更新目标critic
        '''
        # critic 输出的是Q(s,a)这里的a是全局动作
        i = idx
        cur_agent = self.agents[i]
        cur_agent.critic_optimizer.zero_grad()
        # 找出critic的输入
        # 目标网络计算 y=r+ gamma * Critic_target(S_{i+1} , Actor_target( S_{i+1} ) ) ### 目标Q值= r+ gamma * 下一个状态的目标Q值
        # 计算损失 目标Q值 与 当前Q值 之间的MSE
        '''
        下面这一步要保留动作的梯度，
        '''
        all_next_actions = [
            agent_actor(next_state) for agent_actor in self.target_actors
        ]
        target_critic_input = torch.cat((next_state, *all_next_actions), dim=1)

        Q_next_target = cur_agent.target_critic(target_critic_input)

        target_q_values = rewards[i].view(batch_size, -1) + self.gamma * Q_next_target * (1 - done.view(batch_size, -1))

        # 计算当前Q值critic(s,a)
        input_actions = [action.view(batch_size, -1) for action in actions]  # 整理记忆
        input_actions = torch.cat(input_actions, dim=1)
        critic_input = torch.cat((state, input_actions), dim=1)
        q_values = cur_agent.critic(critic_input)
        critic_loss = F.mse_loss(q_values, target_q_values.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        '''
        更新智能体i的actor
        agent_i actor策略的梯度是  agent_i actor动作的梯度 * 全体actor的 Q(s,a1,a2,a3, a_i )的梯度，然后计算均值
        这里的a_i是一个确定性策略 actor_i(state)的输出
        '''
        # 先清零梯度
        cur_agent.actor_optimizer.zero_grad()

        cur_agent_action = cur_agent.actor(state)  # 这个动作需要可导

        all_agent_actions = []
        for idx, agent_actor in enumerate(self.actors):
            if idx == i:
                all_agent_actions.append(cur_agent_action)
            else:
                all_agent_actions.append(agent_actor(state))  # 其他智能体的动作不需要计算梯度
        critic_value_function_in = torch.cat((state, *all_agent_actions), dim=1)
        actor_loss = -cur_agent.critic(critic_value_function_in).mean()  # 一个智能体的策略的更新的原函数是Q_i(s,a1,a2,...,a_i,an)

        actor_loss.backward()
        cur_agent.actor_optimizer.step()

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
            agent.actor.torch.load(name).to(self.device)
