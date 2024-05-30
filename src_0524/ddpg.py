# --------------------------------------------------
# 文件名: ddpg
# 创建时间: 2024/5/14 22:57
# 描述: 原始ddpg方案，直接输出动作
# 作者: WangYuanbo
# --------------------------------------------------
import copy

import torch
import torch.nn.functional as F


def compare_weights(model_before, model_after, model_name):
    print("=========Comparing weights for {}=======".format(model_name))
    for (k1, v1), (k2, v2) in zip(model_before.items(), model_after.items()):
        assert k1 == k2
        print(f"For parameter {k1}, change is {(v2.float() - v1.float()).norm().item()}")


class ActorNet(torch.nn.Module):
    """
    Actor 网络有两个输出，分别是placing动作，和routing动作
    """

    def __init__(self, state_dim, action_dims, hidden_dim):
        super().__init__()
        placing_dim, routing_dim = action_dims
        self.placing_dim = placing_dim
        self.routing_dim = routing_dim
        # self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )

        self.placing_layer = [torch.nn.Linear(hidden_dim, placing_dim[1]) for _ in range(placing_dim[0])]
        self.placing_layer = torch.nn.ModuleList(self.placing_layer)

        self.routing_layer = [torch.nn.Linear(hidden_dim, routing_dim[1]) for _ in range(routing_dim[0])]
        self.routing_layer = torch.nn.ModuleList(self.routing_layer)

    def forward(self, state):
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        x = self.fc(state)
        placing_output = [F.sigmoid(placing_layer(x)) for placing_layer in self.placing_layer]
        # 所有的输出stack起来
        placing_output = torch.stack(placing_output, dim=-2)
        routing_output = [routing_layer(x)for routing_layer in self.routing_layer]
        routing_output = torch.stack(routing_output, dim=-2)
        assert (placing_output.shape[-2], placing_output.shape[-1]) == self.placing_dim, \
            'placing output shape: {}'.format(placing_output.shape)
        assert (routing_output.shape[-2], routing_output.shape[-1]) == self.routing_dim, \
            'routing output shape: {}'.format(routing_output.shape)

        return placing_output, routing_output


class CriticNet(torch.nn.Module):
    def __init__(self, state_dim, action_dims, hidden_dim):
        super().__init__()
        placing_dim, routing_dim = action_dims
        # placing_dim server_num * container_num [0,1]
        # routing_dim user_num * container_num
        placing_dim = placing_dim[0] * placing_dim[1]
        routing_dim = routing_dim[0] * routing_dim[1]
        self.fc1 = torch.nn.Linear(state_dim + placing_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(state_dim + routing_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(placing_dim + routing_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state, actions):
        batch_size = state.shape[0]
        placing_action, routing_action = actions
        assert placing_action.shape[0] == batch_size, 'placing action shape : {}'.format(placing_action.shape)
        assert routing_action.shape[0] == batch_size, 'routing action shape : {}'.format(routing_action.shape)
        placing_action = placing_action.view(batch_size, -1)
        routing_action = routing_action.view(batch_size, -1)
        x1 = torch.cat([state, placing_action], dim=1)
        x2 = torch.cat([state, routing_action], dim=1)
        x3 = torch.cat([placing_action, routing_action], dim=1)
        x1 = F.relu(self.fc1(x1))
        x2 = F.relu(self.fc2(x2))
        x3 = F.relu(self.fc3(x3))
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
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
        placing_params = list(self.actor.fc.parameters()) + list((self.actor.placing_layer.parameters()))
        routing_params = list(self.actor.fc.parameters()) + list((self.actor.routing_layer.parameters()))

        self.placing_actor_optimizer = torch.optim.Adam(placing_params, lr=actor_lr)
        self.routing_actor_optimizer = torch.optim.Adam(routing_params, lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.log_dir = log_dir

    def take_action(self, state, explore: bool):
        self.actor.eval()
        placing_action, routing_action = self.actor(state)
        placing_action = placing_action.detach().squeeze(0)
        routing_action = routing_action.detach().squeeze(0)
        raw_actions = (placing_action, routing_action)
        if explore:
            # 给动作增加噪声
            mu = 0.5
            sigma = 0.2
            placing_action = placing_action + torch.randn_like(placing_action) * sigma + mu
            placing_action = torch.clamp(placing_action, 0, 1)
            routing_action = F.gumbel_softmax(routing_action, dim=-1)
        placing_action = (placing_action > 0.5).int()
        routing_action = torch.argmax(routing_action, dim=-1).int()
        env_action = {
            'placing_action': placing_action,
            'routing_action': routing_action
        }
        return raw_actions, env_action

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
        # self.actor_optimizer.zero_grad()
        # actor_loss = -torch.mean(self.critic(state, self.actor(state)))
        # actor_loss.backward()
        # self.actor_optimizer.step()

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

    def save(self, name):
        torch.save(self.actor, name)


if __name__ == '__main__':
    state_dim = 100
    batch_size = 32
    placing_dim = (2, 3)
    routing_dim = (3, 3)
    action_dims = (placing_dim, routing_dim)

    state = torch.rand((batch_size, state_dim), )
    next_state = torch.rand((batch_size, state_dim), )
    reward = torch.rand((batch_size, 1), )
    gamma = 0.95
    placing_action = torch.rand((batch_size, *placing_dim))
    routing_action = torch.rand((batch_size, *routing_dim))
    action = (placing_action, routing_action)

    actor = ActorNet(state_dim, action_dims, 128)
    actor.train()

    critic = CriticNet(state_dim, action_dims, 128)
    # print(critic)
    actor_before = copy.deepcopy(actor.state_dict())
    critic_before = copy.deepcopy(critic.state_dict())

    target_q_value = reward + gamma * critic(next_state, actor(next_state))
    q_value = critic(state, action)
    critic_loss = F.mse_loss(q_value, target_q_value)
    critic_loss.backward()
    critic_optimizer = torch.optim.Adam(critic.parameters())
    critic_optimizer.step()

    actor_loss = torch.mean(critic(state, actor(state)))
    actor_loss.backward()
    optimizer = torch.optim.Adam(actor.parameters())
    optimizer.step()
    actor_after = copy.deepcopy(actor.state_dict())
    critic_after = copy.deepcopy(critic.state_dict())
    compare_weights(actor_before, actor_after, 'actor')
    compare_weights(critic_before, critic_after, 'critic')


    # batch_size = 10
    # num_features = 20
    # data = torch.randn(batch_size, num_features)
    #
    # # 创建一个BatchNorm1d层, num_features是你的特性数量
    # bn = torch.nn.BatchNorm1d(num_features)
    #
    # # 应用batch normalization
    # normalized_data = bn(data)
    # print(normalized_data.shape)
