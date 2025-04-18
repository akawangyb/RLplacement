# --------------------------------------------------
# 文件名: ppo
# 创建时间: 2025/3/30 20:46
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli

from env_joint_place_distribution import EdgeDeploymentEnv

env = EdgeDeploymentEnv('../data/exptest')
# -------------------------- 环境参数 --------------------------
STATE_DIM = env.state_dim  # 状态维度（如服务器资源状态、请求队列等）
NUM_SERVERS = env.servers  # 边缘服务器数量
NUM_CONTAINERS = env.containers  # 容器类型数量
# 二维0，1
ALLOC_ACTION_DIM = env.max_requests * (env.servers_number + 1)  # 分配动作维度（服务器+云端）

# -------------------------- 超参数 --------------------------
GAMMA = 0.99  # 折扣因子
LAMBDA = 0.95  # GAE参数
CLIP_EPS = 0.2  # PPO剪切阈值
LR_ACTOR = 5e-5  # 策略网络学习率
LR_CRITIC = 5e-5  # 价值网络学习率
BATCH_SIZE = 64  # 训练批次大小
EPOCHS = 10  # 每个数据集的训练轮次
MAX_GRAD_NORM = 0.5  # 梯度裁剪阈值


# -------------------------- 神经网络定义 --------------------------
class ActorCritic(nn.Module):
    """ 策略-价值联合网络，处理多离散动作空间 """

    def __init__(self):
        super().__init__()
        # 共享特征提取层
        self.shared_layer = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # 加载动作头（二值）
        self.load_layer = [nn.Linear(128, env.containers_number)
                           for _ in range(env.servers_number)]
        self.load_head = torch.nn.ModuleList(self.load_layer)
        # 价值函数头
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, alloc_mask=None, load_mask=None):
        # 共享特征
        features = self.shared_layer(x)
        # 加载动作概率（应用Mask）
        load_probs = [torch.sigmoid(layer(features)) for layer in self.load_head]
        load_probs = torch.stack(load_probs, dim=-2)
        if load_mask is not None:
            load_probs = load_probs * load_mask.float()
        # 状态价值
        value = self.value_head(features)
        assert (load_probs.shape[-2], load_probs.shape[-1]) == (env.servers_number, env.containers_number), \
            f"shape not {load_probs.shape}"
        return load_probs, value.squeeze(-1)


# -------------------------- 经验回放缓冲区 --------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, load_action, reward, next_state, done, old_load_logp):
        """ 存储单步经验 """
        self.buffer.append((
            state, load_action, reward, next_state, done,
            old_load_logp
        ))

    def sample(self, batch_size):
        """ 随机采样批次数据 """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        # 解包为张量
        states, load_acts, rewards, next_states, dones, old_load_logps = zip(*batch)
        return (
            torch.FloatTensor(np.stack(states)).view(batch_size, -1),
            torch.FloatTensor(np.stack(load_acts)),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.stack(next_states)),
            torch.FloatTensor(dones),
            torch.stack(old_load_logps, dim=0)
        )

    def __len__(self):
        return len(self.buffer)


# -------------------------- 优势计算函数 --------------------------
def compute_gae(rewards, values, next_values, dones, gamma=GAMMA, lambda_=LAMBDA):
    """ 计算广义优势估计 (GAE) """
    batch_size = len(rewards)
    advantages = np.zeros(batch_size)
    last_gae = 0

    for t in reversed(range(batch_size)):
        if dones[t]:
            delta = rewards[t] - values[t]
            last_gae = delta
        else:
            delta = rewards[t] + gamma * next_values[t] - values[t]
            last_gae = delta + gamma * lambda_ * last_gae
        advantages[t] = last_gae
    return torch.FloatTensor(advantages)


# -------------------------- 训练流程 --------------------------
class PPO:
    def __init__(self):
        self.net = ActorCritic()
        self.optimizer = optim.Adam([
            {'params': self.net.shared_layer.parameters(), 'lr': LR_ACTOR},
            {'params': self.net.load_head.parameters(), 'lr': LR_ACTOR},
            {'params': self.net.value_head.parameters(), 'lr': LR_CRITIC}
        ])
        self.buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state, load_mask):
        """ 使用当前策略选择动作 """
        with torch.no_grad():
            # 前向传播并应用Mask
            load_probs, value = self.net(
                torch.FloatTensor(state), None, None
                # torch.FloatTensor(state).unsqueeze(0), None, None
                # alloc_mask=torch.BoolTensor(alloc_mask).unsqueeze(0) ,
                # load_mask=torch.BoolTensor(load_mask).unsqueeze(0)
            )
            # 分配动作采样
            # probs表示已经经过归一化

            # 加载动作采样
            load_dist = Bernoulli(probs=load_probs)
            load_action = load_dist.sample()
            load_logp = load_dist.log_prob(load_action)
            load_logp = torch.sum(load_logp, dim=-1)
        return (load_action, load_logp, value.item())

    def update(self):
        """ 执行PPO策略更新 """
        if len(self.buffer) < BATCH_SIZE:
            return

        for _ in range(EPOCHS):
            # 采样批次数据
            states, load_acts, rewards, next_states, dones, old_load_logps = \
                self.buffer.sample(BATCH_SIZE)

            assert states.shape == (BATCH_SIZE, env.state_dim), \
                f"shape not {states.shape}"
            assert load_acts.shape == (BATCH_SIZE, env.servers_number, env.containers_number), \
                f"shape not {states.shape}"

            # 计算新策略的值和动作概率
            load_probs, values = self.net(states)
            next_load_probs, next_values = self.net(next_states)
            # 加载动作对数概率
            load_dist = Bernoulli(probs=load_probs)
            new_load_logps = load_dist.log_prob(load_acts).sum(dim=-1)

            # 重要性采样比率
            ratios_load = torch.exp(new_load_logps - old_load_logps)
            ratios_load = torch.sum(ratios_load, dim=-1)
            ratios = ratios_load

            # 计算优势函数
            advantages = compute_gae(rewards.numpy(), values.detach().numpy(),
                                     next_values.detach().numpy(), dones.numpy())
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 策略损失 (PPO-CLIP)
            assert ratios.shape == advantages.shape, f'not shape ratios{ratios.shape} advantage{advantages.shape}'
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值函数损失
            returns = advantages + values.detach()
            value_loss = F.mse_loss(values, returns)

            # 熵正则化
            entropy_load = load_dist.entropy().mean()
            entropy_loss = -0.01 * (entropy_load)

            # 总损失
            total_loss = policy_loss + value_loss + entropy_loss

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()


def solve_the_alloc_act(load_act: torch.Tensor, request: list):
    load_act = load_act.cpu().detach().numpy()
    alloc_act = np.zeros(env.max_requests)

    # 默认全部由云完成服务
    for rs in range(len(request)):
        alloc_act[rs] = env.servers_number
    server_info = copy.deepcopy(env.servers)
    # 基于贪心分配
    # 对服务请求实现贪心排序
    # 0.5 0.1 .0.2 0.2
    k1 = 0.5
    k2 = 0.1
    k3 = 0.2
    k4 = 0.2
    sorted_request = sorted(request,
                            key=lambda x: k1 * x['cpu_demand'] + \
                                          k2 * x['mem_demand'] + \
                                          k3 * x['upload_demand'] + \
                                          k4 * x['download_demand'])
    # 这一步的目的只是获得一个可用的解
    request_server_set = []
    server_process_set = [set() for _ in range(len(request))]

    # for server_id in range(env.servers):
    #     _can_do_request_set = set()
    #     for id, request in enumerate(sorted_request):
    #         container_name = request['service_type']
    #         container_id = env.service_type_to_int(container_name)
    #         if load_act[server_id][container_id] == 1:
    #             _can_do_request_set.add(id)
    #         # 当前服务完成该请求，减去相应的资源
    #         if server_info[server_id]['cpu_capacity'] > request['cpu_demand'] and \
    #                 server_info[server_id]['mem_capacity'] > request['mem_demand'] and \
    #                 server_info[server_id]['upload_capacity'] > request['upload_demand'] and \
    #                 server_info[server_id]['download_capacity'] > request['download_demand']:
    #
    #             server_info[server_id]['cpu_capacity'] -= request['cpu_demand']
    #             server_info[server_id]['mem_capacity'] -= request['mem_demand']
    #             server_info[server_id]['upload_capacity'] -= request['upload_demand']
    #             server_info[server_id]['download_capacity'] -= request['download_demand']
    #         else:
    #             alloc_act[]
    #             server_info[server_id]['cpu_capacity'] -= server_info[server_id]

    for r_id, r in enumerate(request):
        t_set = set()
        for n in range(env.servers_number):
            container_name = r['service_type']
            container_id = env.service_type_to_int[container_name]
            if load_act[n][container_id] == 1:
                t_set.add(container_id)
                server_process_set[r_id].add(r_id)
                server_id = n
                # 判断资源 当前服务完成该请求，减去相应的资源
                if server_info[server_id]['cpu_capacity'] > r['cpu_demand'] and \
                        server_info[server_id]['mem_capacity'] > r['mem_demand'] and \
                        server_info[server_id]['upload_capacity'] > r['upload_demand'] and \
                        server_info[server_id]['download_capacity'] > r['download_demand']:
                    alloc_act[r_id] = n
                    server_info[server_id]['cpu_capacity'] -= r['cpu_demand']
                    server_info[server_id]['mem_capacity'] -= r['mem_demand']
                    server_info[server_id]['upload_capacity'] -= r['upload_demand']
                    server_info[server_id]['download_capacity'] -= r['download_demand']
        request_server_set.append(t_set)

    return torch.Tensor(alloc_act)


# -------------------------- 示例用法 --------------------------
if __name__ == "__main__":
    agent = PPO()
    # 模拟环境交互
    for episode in range(1000):
        total_reward = 0
        total_ppo_reward = 0
        invalid_actions=0
        state, done = env.reset()
        while not done:
            load_mask = env.get_loading_mask()
            # 选择动作
            load_act, load_logp, value = agent.select_action(state, load_mask)
            assert load_act.shape == (env.servers_number, env.containers_number), \
                f"not shape {load_act.shape}"
            alloc_act = solve_the_alloc_act(load_act, env.current_requests)
            # if episode == 1:
            #     print(load_act)

            # 根据load动作计算alloc动作
            # 模拟环境返回奖励和新状态
            action = (alloc_act, load_act)
            assert load_act.shape == (env.servers_number, env.containers_number), f"shape not {load_act.shape}"
            next_state, reward, done, info = env.step(action)
            ppo_reward = info['container_reward'] * 0.4 + info['prop_reward'] * 0.4 + info['compute_reward'] * 0.2
            total_ppo_reward += info['container_reward']
            invalid_actions += info['invalid_actions']
            # 存储经验
            agent.buffer.add(state, load_act, -ppo_reward, next_state, done, load_logp)
            # agent.buffer.add(state, load_act, reward, next_state, done, load_logp)
            # total_reward += -ppo_reward
            total_reward += reward
        agent.update()
        print("episode", episode, 'reward', total_reward, 'ppo_reward', total_ppo_reward)
        print("invalid actions", invalid_actions)
