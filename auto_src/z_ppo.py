# --------------------------------------------------
# 文件名: z_ppo
# 创建时间: 2024/8/4 22:38
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import time
from collections import namedtuple

import torch
import torch.nn.functional as F
import yaml
from torch.distributions import Categorical

from auto_src.env_with_interference import CustomEnv
from auto_src.tools import base_opt


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


class PPO:

    def __init__(self, state_dim, action_dim, hidden_dim,
                 actor_lr, critic_lr, gamma, tau, log_dir, device,
                 eps=0.2, lmbda=0.95,epochs=10):
        self.actor = ActorNet(state_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNet(state_dim, action_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.bc_optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=0.00005)
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.log_dir = log_dir
        self.action_dim = action_dim

        self.eps = eps
        self.lmbda = lmbda
        self.epochs = epochs

    def take_action(self, state, explore: bool, eps=0.1):
        self.actor.eval()
        action_probs = self.actor(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        action = F.one_hot(action, num_classes=self.action_dim[1])
        return action

    def update(self, transition_dict):
        """
        值函数的优化目标是最小化 状态的预测价值，预期实际价值之间的差距， 实际价值是通过贝尔曼方程得到的
        q_value= critic(state)
        target_q_value= rewards + self.gamma * self.critic(next_states) * (1 - dones)  环境奖励 + 折扣*下一状态的价值
        :param transition_dict:
        """
        self.actor.train()
        states = transition_dict['states']
        states = torch.concat(states, dim=0)

        actions = transition_dict['actions']  # 是一个list，每个元素是一个字典
        actions = torch.stack(actions, dim=0)

        next_states = transition_dict['next_states']
        next_states = torch.concat(next_states, dim=0)

        rewards = transition_dict['rewards']
        rewards = torch.stack(rewards, dim=0).float().view(-1, 1)
        dones = transition_dict['dones']
        dones = torch.stack(dones, dim=0).int().view(-1, 1)

        batch_size = states.shape[0]
        assert rewards.shape == (batch_size, 1)
        assert rewards.shape == dones.shape
        assert next_states.shape == states.shape
        assert actions.shape[0] == batch_size

        target_q_value = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        q_value = self.critic(states)  # 此处是状态的价值
        td_error = target_q_value - q_value
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_error).to(self.device)
        # 计算旧策略的动作概率分布,只取特定动作的概率分布

        index_actions = torch.argmax(actions, dim=-1).unsqueeze(-1)
        old_probs = torch.gather(self.actor(states), dim=-1, index=index_actions).squeeze(-1)
        log_old_probs = torch.log(old_probs).detach()

        # old_log_probs = torch.log_reserved(
        # torch.gather(self.actor(states), dim=-1, index=index_actions).squeeze(-1).prod(dim=-1).view(-1, 1)).detach()
        for _ in range(self.epochs):
            actor_before = self.actor.state_dict()
            critic_before = self.critic.state_dict()
            # 计算新的动作概率分布
            new_probs = torch.gather(self.actor(states), dim=-1, index=index_actions).squeeze(-1)
            log_new_probs = torch.log(new_probs)
            ratio = torch.exp(log_new_probs - log_old_probs)
            # ratio = torch.prod(ratio, dim=-1).view(-1, 1)

            # log_probs = torch.log_reserved(
            #     torch.gather(self.actor(states), dim=-1, index=index_actions).squeeze(-1).prod(dim=-1).view(-1, 1))
            # ratio = torch.exp(log_probs - old_log_probs)
            # 这里的新旧动作
            assert ratio.shape == advantage.shape, \
                f'ratio shape is {ratio.shape} and advantage shape is {advantage.shape}'
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            assert self.critic(states).shape == target_q_value.shape, \
                f'critic shape is {self.critic(states).shape} and target q value shape is {target_q_value.shape}'
            critic_loss = torch.mean(F.mse_loss(self.critic(states), target_q_value.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            # torch.nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=0.1)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            actor_after = self.actor.state_dict()
            critic_after = self.critic.state_dict()
            # print_model_parameters(self.actor, 'actor')
            # print_model_parameters(self.critic, 'critic')
            # compare_weights(actor_before, actor_after, 'actor')
            # compare_weights(critic_before, critic_after, 'critic')

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float, device=self.device)


with open('train_config.yaml', 'r', encoding='utf-8') as f:
    config_data = yaml.safe_load(f)

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
                     'tau',
                     'lmbda',
                     'epochs',
                     'eps',
                     'data_dir'
                     ])
config = Config(**config_data)

f_log, log_dir, device = base_opt('env', config.data_dir)

env = CustomEnv(device=device, data_dir=config.data_dir)

state_dim = env.state_dim

action_dim = (env.container_number, env.server_number + 1)
critic_input_dim = state_dim + action_dim[0] * action_dim[1]

agent = PPO(state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            gamma=config.gamma,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            device=device,
            eps=0.2)

last_time = time.time()
return_list = []
for i_episode in range(config.num_episodes):
    state, done = env.reset()  # 这里的state就是一个tensor
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    episode_return = torch.zeros(env.container_number)
    while not done:
        state = state.view(1, -1)
        action = agent.take_action(state)
        next_state, rewards, done, info = env.step(action)
        next_state = next_state.view(1, -1)
        rewards = -rewards
        rewards = torch.sum(rewards)
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['rewards'].append(rewards)
        transition_dict['next_states'].append(next_state)
        transition_dict['dones'].append(done)
        state = next_state
        episode_return += rewards
    return_list.append(episode_return)
    print('Episode {} return: {}'.format(i_episode, torch.sum(episode_return).item()))
    f_log.write('Episode {} return: {} \n'.format(i_episode, episode_return[-1].item()))
    agent.update(transition_dict)
