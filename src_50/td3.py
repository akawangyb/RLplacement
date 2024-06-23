# --------------------------------------------------
# 文件名: td3
# 创建时间: 2024/6/22 15:31
# 描述: 设置2个Q网络
# 作者: WangYuanbo
# --------------------------------------------------
import copy

import torch
import torch.nn.functional as F

from tools import one_hot


def compare_weights(model_before, model_after, model_name):
    print("=========Comparing weights for {}=======".format(model_name))
    for (k1, v1), (k2, v2) in zip(model_before.items(), model_after.items()):
        assert k1 == k2
        print(f"For parameter {k1}, change is {(v2.float() - v1.float()).norm().item()}")


class ActorNet(torch.nn.Module):
    """
    Actor 网络有两个输出，分别是placing动作
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        # action_dim container * server+1
        self.action_dim = action_dim
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.placing_layer = [torch.nn.Linear(hidden_dim, action_dim[1]) for _ in range(action_dim[0])]
        self.placing_layer = torch.nn.ModuleList(self.placing_layer)

    def forward(self, state):
        x = self.fc(state)
        output = [F.relu(placing_layer(x)) for placing_layer in self.placing_layer]
        # 所有的输出stack起来
        output = torch.stack(output, dim=-2)
        assert (output.shape[-2], output.shape[-1]) == self.action_dim, \
            'placing output shape: {}'.format(output.shape)

        return output


class CriticNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        flatten_action_dim = action_dim[0] * action_dim[1]
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim + flatten_action_dim, 2 * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        batch_size = state.shape[0]
        assert action.shape[0] == batch_size, 'action shape : {}'.format(action.shape)
        action = action.view(batch_size, -1)
        x = torch.cat([state, action], dim=1)
        x = self.fc(x)
        return x


class TD3:
    '''
    DDPG算法
    '''

    def __init__(self, state_dim, action_dim, hidden_dim,
                 actor_lr, critic_lr, gamma, tau, log_dir, device):
        self.actor = ActorNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = ActorNet(state_dim, action_dim, hidden_dim).to(device)

        self.critic_1 = CriticNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic_1 = CriticNet(state_dim, action_dim, hidden_dim).to(device)

        self.critic_2 = CriticNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic_2 = CriticNet(state_dim, action_dim, hidden_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.log_dir = log_dir
        self.action_dim = action_dim

    def take_action(self, state, explore: bool, eps=0.1):
        self.actor.eval()
        action = self.actor(state)
        action = action.detach().squeeze(0)
        if explore:
            # 给动作增加噪声
            action = F.gumbel_softmax(action, dim=-1)
            action = one_hot(action, eps=eps)
        else:
            action = one_hot(action, 0)

        return action

    def update(self, batch):
        self.actor.train()
        actor_before, critic_before = copy.deepcopy(self.actor.state_dict()), copy.deepcopy(self.critic_1.state_dict())
        state, action, reward, next_state, done = batch

        # 先更新critic网络

        next_q_value_1 = self.target_critic_1(next_state, self.target_actor(next_state))
        next_q_value_2 = self.target_critic_2(next_state, self.target_actor(next_state))

        next_q_value = torch.min(next_q_value_1, next_q_value_2)
        target_q_value = reward + (self.gamma * next_q_value * (1 - done)).detach()

        # 更新q_1网络
        self.critic_1_optimizer.zero_grad()
        q_value_1 = self.critic_1(state, action)
        assert target_q_value.shape == q_value_1.shape, \
            f'target_q_value.shape={target_q_value.shape} q_value.shape={q_value_1.shape}'
        critic_loss = F.mse_loss(target_q_value, q_value_1)
        critic_loss.backward()
        self.critic_1_optimizer.step()

        # 更新q_2网络
        self.critic_2_optimizer.zero_grad()
        q_value_2 = self.critic_2(state, action)
        assert target_q_value.shape == q_value_2.shape, \
            f'target_q_value.shape={target_q_value.shape} q_value.shape={q_value_2.shape}'
        critic_2_loss = F.mse_loss(target_q_value, q_value_2)
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新actor网络
        self.actor_optimizer.zero_grad()
        actor_loss = -torch.mean(self.critic_1(state, self.actor(state)))
        actor_loss.backward()
        self.actor_optimizer.step()

        actor_after, critic_after = copy.deepcopy(self.actor.state_dict()), copy.deepcopy(self.critic_1.state_dict())
        # compare_weights(actor_before, actor_after, 'actor')
        # compare_weights(critic_before, critic_after, 'critic')

        # 输出actor网络的输出层的参数
        # for k, v in self.actor.placing_layer.state_dict().items():
        #     print(k, v)

        # 软更新
        self.soft_update(self.actor, self.target_actor, self.tau)
        self.soft_update(self.critic_1, self.target_critic_1, self.tau)
        self.soft_update(self.critic_2, self.target_critic_2, self.tau)

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)

    def save(self, name):
        torch.save(self.actor, name)

    def learn(self, transition_dict):
        states = transition_dict['states']
        states = torch.concat(states, dim=0)

        actions = transition_dict['actions']  # 是一个list，每个元素是一个字典
        actions = torch.stack(actions, dim=0)

        next_states = transition_dict['next_states']
        next_states = torch.concat(next_states, dim=0)

        rewards = transition_dict['rewards']
        # discounted_rewards = []
        # # 把每一个奖励变成其实际奖励
        # last_reward = 0
        # for reward in reversed(rewards):
        #     discounted_rewards.append(reward + self.gamma * last_reward)
        #     last_reward = reward
        # rewards = reversed(discounted_rewards)
        # rewards = torch.Tensor(rewards).view(-1, 1)
        rewards = torch.stack(rewards, dim=0).float().view(-1, 1)

        dones = transition_dict['dones']
        dones = torch.stack(dones, dim=0).int().view(-1, 1)

        batch_size = states.shape[0]
        assert rewards.shape == (batch_size, 1), f'rewards shape {rewards.shape} does not match batch size {batch_size}'
        assert rewards.shape == dones.shape
        assert next_states.shape == states.shape
        assert actions.shape[0] == batch_size
        # 假设只先学习actor
        for _ in range(20):
            self.actor.train()
            dim = actions.shape[-1]
            y = F.softmax(self.actor(states), dim=-1).view(-1, dim)
            target_action = actions.view(-1, dim)
            target_action = torch.argmax(target_action, dim=-1)
            actor_loss = torch.nn.CrossEntropyLoss()(y, target_action)
            actor_before = self.actor.state_dict()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        self.target_actor.load_state_dict(self.actor.state_dict())

        # # 假设学习critic网络
        # for _ in range(10):
        #     self.critic_optimizer.zero_grad()
        #     critic_loss = F.mse_loss(self.critic(states, actions), rewards)
        #     critic_loss.backward()
        #     self.critic_optimizer.step()
        # actor_after = self.actor.state_dict()
        # compare_weights(actor_before, actor_after, 'actor')
