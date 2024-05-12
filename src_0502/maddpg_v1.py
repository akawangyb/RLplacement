# --------------------------------------------------
# 文件名: maddpg
# 创建时间: 2024/5/2 12:07
# 描述:  maddpg算法描述,只考虑2个智能体
# 作者: WangYuanbo
# --------------------------------------------------
import os

import torch
import torch.nn.functional as F

from tools import onehot_from_logits, gumbel_softmax


class RoutingNet(torch.nn.Module):
    def __init__(self, num_in, hidden_dim, user_number, container_number):
        super().__init__()
        self.user_number = user_number
        self.container_number = container_number
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = [torch.nn.Linear(hidden_dim, container_number)] * self.user_number
        self.fc3 = torch.nn.ModuleList(self.fc3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = []
        for i in range(self.user_number):
            output.append(F.softmax(self.fc3[i](x), dim=-1))
        x = torch.stack(output, dim=1)

        return x


class PlacingNet(torch.nn.Module):
    def __init__(self, num_in, hidden_dim, server_number, container_number):
        super().__init__()
        self.server_number = server_number
        self.container_number = container_number
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = [[torch.nn.Linear(hidden_dim, 2)] * self.container_number] * self.server_number
        self.fc3 = torch.nn.ModuleList([torch.nn.ModuleList(layer) for layer in self.fc3])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        output = []
        for i in range(self.server_number):
            temp_tensor = []
            for j in range(self.container_number):
                temp_tensor.append(F.sigmoid(self.fc3[i][j](x)))
            output.append(torch.stack(temp_tensor, dim=1))  # stack是新增加一个维度
        x = torch.stack(output, dim=1)  # shape server_number*container_number*2
        return x


class CriticNet(torch.nn.Module):
    def __init__(self, dims, hidden_dim=128):
        super().__init__()
        state_dim, placing_action_dim, routing_action_dim = dims
        self.state_dim = state_dim
        self.placing_action_dim = placing_action_dim
        self.routing_action_dim = routing_action_dim
        self.hidden_dim = hidden_dim

        self.placing_net = torch.nn.Sequential(
            torch.nn.Linear(placing_action_dim[0] * placing_action_dim[1] * placing_action_dim[2],
                            placing_action_dim[0] * placing_action_dim[1]),
            torch.nn.ReLU()
        )
        self.routing_net = torch.nn.Sequential(
            torch.nn.Linear(routing_action_dim[0] * routing_action_dim[1], routing_action_dim[0]),
            torch.nn.ReLU()
        )

        self.merge_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim + placing_action_dim[0] * placing_action_dim[1] + routing_action_dim[0],
                            hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        state, placing_action, routing_action = x
        batch_size = state.shape[0]
        assert batch_size == placing_action.shape[0], \
            f'state shape is {state.shape}, placing_action shape is {placing_action.shape}'
        assert batch_size == routing_action.shape[0], \
            f'state shape is {state.shape}, routing_action shape is {routing_action.shape}'
        placing_action = placing_action.view(batch_size, -1)
        routing_action = routing_action.view(batch_size, -1)
        placing_x = self.placing_net(placing_action)
        routing_x = self.routing_net(routing_action)
        merge_x = torch.cat([state, routing_x, placing_x], dim=-1)
        x = self.merge_net(merge_x)
        return x


class RoutingDDPG:
    '''
     DDPG算法
     '''

    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):

        self.actor = RoutingNet(state_dim, hidden_dim, action_dim[0], action_dim[1]).to(device)
        self.target_actor = RoutingNet(state_dim, hidden_dim, action_dim[0], action_dim[1]).to(device)

        self.critic = CriticNet(critic_input_dim, hidden_dim).to(device)
        self.target_critic = CriticNet(critic_input_dim, hidden_dim).to(device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state, explore=False):
        action = self.actor(state)
        if explore:
            action = gumbel_softmax(action)

        else:
            action = onehot_from_logits(action)  # 这里的代码要改

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
        self.actor = PlacingNet(state_dim, hidden_dim, action_dim[0], action_dim[1]).to(device)
        self.target_actor = PlacingNet(state_dim, hidden_dim, action_dim[0], action_dim[1]).to(device)
        self.critic = CriticNet(critic_input_dim).to(device)
        self.target_critic = CriticNet(critic_input_dim).to(device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state, explore=False):
        """
        重写此处
        :param state:
        :param explore: 探索时，动作需要有梯度，但是开发时，动作不需要梯度
        :return:
        """
        action = self.actor(state)

        if explore:
            # # 添加噪声,进行随机探索
            # noise = torch.randn_like(action)  # 0,1
            # noise = noise * 0.1 + 0.5  # 假设噪声的分布式服从 N(0.5,0.1)
            # action += noise
            # action = torch.clamp(action, min=0, max=1)
            print('explore before processed')
            print(action, action.shape)
            action = gumbel_softmax(action)
            print('after gumbel')
            print(action, action.shape)
        else:
            action = onehot_from_logits(action)
        # # 如果不是探索直接利用action

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

        self.placing_agents_number = env.placing_agents_number
        self.routing_agents_number = env.routing_agents_number

        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        self.container_number = env.container_number
        self.log_dir = log_dir

        self.agents.append(
            PlacingDDPG(state_dims[0], action_dims[0], critic_input_dim,
                        hidden_dim, actor_lr, critic_lr, device))

        self.agents.append(
            RoutingDDPG(state_dims[1], action_dims[1], critic_input_dim,
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

        placing_action = self.agents[0].take_action(state, explore)
        routing_action = self.agents[1].take_action(state, explore)

        raw_actions = (placing_action, routing_action)
        # 这里把动作转换成env_action
        placing_action = torch.argmax(placing_action, dim=-1).int().squeeze(0)
        routing_action = torch.argmax(routing_action, dim=-1).int().squeeze(0)
        # print('placing_action', placing_action)
        # print('routing_action', routing_action)

        # if explore:
        # placing_action = placing_action.squeeze(0)
        # routing_action = routing_action.squeeze(0)
        env_action = {
            'placing_action': placing_action,
            'routing_action': routing_action
        }
        return raw_actions, env_action

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
        # 目标网络计算 y=r+ gamma * Critic_target( S_{i+1} , Actor_target( S_{i+1} ) ) ### 目标Q值= r+ gamma * 下一个状态的目标Q值
        # 计算损失 目标Q值 与 当前Q值 之间的MSE
        '''
        下面这一步要保留动作的梯度，
        '''

        placing_action, routing_action = self.agents[0].target_actor(next_state), self.agents[1].target_actor(
            next_state)
        target_critic_input = (next_state, placing_action, routing_action)
        Q_next_target = cur_agent.target_critic(target_critic_input)

        target_q_values = rewards[i].view(batch_size, -1) + self.gamma * Q_next_target * (
                1 - done.view(batch_size, -1))

        # 计算当前Q值critic(s,a)
        critic_input = (state, actions[0], actions[1])  # 这里需要把所有记忆的智能体动作取出来
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
        critic_value_function_in = (state, all_agent_actions[0], all_agent_actions[1])
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


if __name__ == '__main__':
    net = PlacingNet(100, 128, 3, 10)
    # net = RoutingNet(100, 128, 3, 10)
    batch_size = 256
    num_in = 100
    action = torch.zeros(batch_size, num_in)
    res = net(action)
    print(res.shape)
