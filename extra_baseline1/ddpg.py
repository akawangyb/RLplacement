# --------------------------------------------------
# 文件名: ddpg
# 创建时间: 2024/6/5 15:25
# 描述:
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


class DDPG:
    '''
    DDPG算法
    '''

    def __init__(self, state_dim, action_dim, hidden_dim,
                 actor_lr, critic_lr, gamma, tau, log_dir, device):
        self.actor = ActorNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = ActorNet(state_dim, action_dim, hidden_dim).to(device)

        self.critic = CriticNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = CriticNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.bc_optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=0.0005)
        self.bc_optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=0.003)
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
        actor_before, critic_before = copy.deepcopy(self.actor.state_dict()), copy.deepcopy(self.critic.state_dict())
        state, action, reward, next_state, done = batch

        # 先更新critic网络
        self.critic_optimizer.zero_grad()
        next_q_value = self.target_critic(next_state, self.target_actor(next_state))
        target_q_value = reward + self.gamma * next_q_value * (1 - done)
        q_value = self.critic(state, action)
        assert target_q_value.shape == q_value.shape, \
            f'target_q_value.shape={target_q_value.shape} q_value.shape={q_value.shape}'
        critic_loss = F.mse_loss(target_q_value, q_value)

        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新actor网络

        self.actor_optimizer.zero_grad()
        actor_loss = -torch.mean(self.critic(state, self.actor(state)))
        actor_loss.backward()
        self.actor_optimizer.step()

        actor_after, critic_after = copy.deepcopy(self.actor.state_dict()), copy.deepcopy(self.critic.state_dict())
        # compare_weights(actor_before, actor_after, 'actor')
        # compare_weights(critic_before, critic_after, 'critic')

        # 输出actor网络的输出层的参数
        # for k, v in self.actor.placing_layer.state_dict().items():
        #     print(k, v)

        # 软更新
        self.soft_update(self.actor, self.target_actor, self.tau)
        self.soft_update(self.critic, self.target_critic, self.tau)

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)

    def save(self, name='actor_parameters.pth'):
        torch.save(self.actor.state_dict(), name)

    def learn(self, transition_dict, epoch=10, critic_learn=-1):
        states = transition_dict['states']
        states = torch.concat(states, dim=0)

        actions = transition_dict['actions']  # 是一个list，每个元素是一个字典
        actions = torch.stack(actions, dim=0)

        next_states = transition_dict['next_states']
        next_states = torch.concat(next_states, dim=0)

        rewards = transition_dict['rewards']

        dones = transition_dict['dones']
        dones = torch.stack(dones, dim=0).int().view(-1, 1)

        batch_size = states.shape[0]

        assert next_states.shape == states.shape
        assert actions.shape[0] == batch_size
        # 假设只先学习actor
        for _ in range(epoch):
            self.actor.train()
            dim = actions.shape[-1]
            y = F.softmax(self.actor(states), dim=-1).view(-1, dim)
            target_action = actions.view(-1, dim)
            target_action = torch.argmax(target_action, dim=-1)
            actor_loss = torch.nn.CrossEntropyLoss()(y, target_action)
            self.bc_optimizer_actor.zero_grad()
            actor_loss.backward()
            self.bc_optimizer_actor.step()
        self.target_actor.load_state_dict(self.actor.state_dict())
        if critic_learn == -1:
            return
        discounted_rewards = []
        # 把每一个奖励变成其实际奖励
        last_reward = torch.zeros_like(rewards[0])
        for reward in reversed(rewards):
            discounted_rewards.append(reward + self.gamma * last_reward)
            last_reward = discounted_rewards[-1]

        rewards = discounted_rewards[::-1]
        rewards = torch.stack(rewards, dim=0).view(-1, 1)
        assert rewards.shape == (batch_size, 1), f"rewards shape {rewards.shape} does not match"

        # 假设学习critic网络
        for _ in range(critic_learn):
            critic_loss = F.mse_loss(self.critic(states, actions), rewards)
            print("epoch ", _, "critic loss", critic_loss.item())
            self.bc_optimizer_critic.zero_grad()
            critic_loss.backward()
            self.bc_optimizer_critic.step()
        self.target_critic.load_state_dict(self.critic.state_dict())
