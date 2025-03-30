# --------------------------------------------------
# 文件名: ppo_edge_place
# 创建时间: 2025/3/29 19:48
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import copy
import random
from collections import deque

import torch
from torch import nn, optim

from env_joint_place_distribution import EdgeDeploymentEnv


def get_placement(action):
    routing_action = torch.argmax(action, dim=-1).tolist()
    x = [[0] * (env.servers_number + 1)] * env.max_requests
    for id, r in enumerate(routing_action):
        x[id][r] = 1
    # 根据x来求y
    y = [[0] * env.containers_number] * env.servers_number

    requests = env.request_generator.generate(env.current_timestep)
    # 遍历所有的服务器
    for n in range(env.servers_number):
        sum_d = 0
        required_containers = set()
        # for routing_id, r in enumerate(routing_action):
        for routing_id, r in enumerate(requests):
            if r == n:  # 应该以在线的方式部署容器
                container_id = env.service_type_to_int[r['service_type']]
                required_containers.add(container_id)
                sum_d += r['mem_usage']
        sum_h = 0
        for container_id in required_containers:
            container_name = env.containers[container_id]
            for r in requests:
                if r['service_type'] == container_name:
                    sum_h += r['h_c']
                    break
        total_mem = sum_d + sum_h
        # 校验内存约束
        if total_mem > env.servers[n]['mem_capacity']:
            raise ValueError(f"服务器 {n} 内存不足（需{total_mem}）")
        # 设置必须加载的容器
        for c in required_containers:
            y[n][c] = 1
    for n in range(env.servers_number):
        sum_d = 0
        for c in range(env.containers_number):
            sum_d += env.h_c_map[env.containers[c]]
            if y[n][c] == 1:
                continue
            if sum_d <= env.servers[n]['mem_capacity']:
                y[n][c] = 1
    return y


# ------------------- 神经网络定义 -------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, num_servers, max_requests):
        super().__init__()
        self.num_servers = num_servers
        # self.num_containers = num_containers
        self.max_requests = max_requests
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # 容器部署分支（服务器×容器）
        # self.container_deploy = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, num_servers * num_containers),
        #     nn.Sigmoid()  # 输出概率
        # )

        # 请求分配分支（动态候选）
        self.request_alloc = nn.Sequential(
            nn.Linear(128, 256),  # 共享特征+请求特征
            nn.ReLU(),
            nn.Linear(256, max_requests * (self.num_servers + 1))  # 假设最大候选数=服务器数+1（云）
        )

        # 价值网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        shared_feat = self.shared_layer(state)
        # 容器部署概率
        container_probs = self.container_deploy(shared_feat).view(-1, self.num_servers, self.num_containers)
        # 请求分配概率
        # request_feat = self._encode_requests(requests)
        # alloc_logits = self.request_alloc(torch.cat([shared_feat, request_feat], dim=1))
        print(shared_feat.shape)
        alloc_logits = self.request_alloc(shared_feat)
        return container_probs, alloc_logits, self.critic(state)

    def _encode_requests(self, requests):
        """编码请求特征（示例实现）"""
        return torch.mean(requests, dim=1)  # 实际可用Transformer替换


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, container_act, request_act, reward, next_state, done):
        self.buffer.append((state, container_act, request_act, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class PPOAgent:
    def __init__(self, config):
        self.server_number = config['num_servers']
        self.container_number = config['num_containers']
        self.device = config['device']
        self.gamma = config['gamma']
        self.eps_clip = config['eps_clip']
        self.K_epochs = config['K_epochs']
        self.max_requests = config['max_requests']

        self.policy = ActorCritic(config['state_dim'],
                                  config['num_servers'],
                                  config['max_requests']).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['lr'])
        self.buffer = ReplayBuffer(config['buffer_capacity'])

        self.old_policy = ActorCritic(config['state_dim'],
                                      config['num_servers'],
                                      config['max_requests']).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())

    def trans_state(self, state):
        """构建固定长度的状态向量"""
        # mem_state = state['server_mem_available']
        # cpu_state = state['server_cpu_usage']
        # request = state['requests']
        # new_state = []
        # new_state += cpu_state.tolist()
        # new_state += mem_state.tolist()
        # # 处理实际请求（最多取前 max_requests 个）
        # valid_requests = request
        # # 编码有效请求的特征
        # for req in valid_requests:
        #     new_state.extend([
        #         req['cpu_demand'],
        #         req['mem_demand'],
        #         req['upload_demand'],
        #         req['download_demand'],
        #         req['h_c']
        #     ])
        # # 填充零值（如果请求数不足 max_requests）
        # padding_length = (self.max_requests - len(valid_requests)) * 5  # 每个请求5个特征
        # print('new_state', new_state)
        # new_state += [0.0] * padding_length
        #
        # # 添加其他状态信息（如服务器状态、时隙等）
        # new_state.append(state['time_step'])
        # print(new_state)
        # return torch.tensor(new_state, dtype=torch.float32)
        return torch.tensor(state, dtype=torch.float32)

    def act(self, state):
        # 环境状态是一个字典，把他转换转换为动作网络的输入
        # trans_tensor_state = self.trans_state(state)
        with torch.no_grad():
            # 生成容器部署掩码（示例：内存限制）
            # 添加掩码动态过滤无效动作，确保智能体不会选择导致内存超载的容器部署动作
            mem_available = state['server_mem_available']
            # print(trans_tensor_state.shape)
            container_probs = self.old_policy.container_deploy(self.old_policy.shared_layer(state))
            container_probs = container_probs.view(self.server_number, self.container_number)
            mem_mask = torch.Tensor(mem_available).unsqueeze(1)  # 应用掩码
            # 正确示例：根据可用内存生成0/1掩码
            mask = (server_mem_available >= container_mem_demand).float()  # 结果只能是0或1
            # print(mem_mask, mem_mask.shape)
            # print(container_probs, container_probs.shape)
            container_probs *= mem_mask  # 应用掩码
            # 采样动作
            # 生成二项分布结果
            container_act = torch.bernoulli(container_probs)
            request_act = self._sample_requests(state)

        return container_act, request_act

    def _sample_requests(self, container_act, requests):
        """动态生成候选服务器并采样请求分配"""
        valid_servers = [i for i, s in enumerate(container_act) if s.sum() > 0]
        candidates = valid_servers + [self.server_number]
        return [random.choice(candidates) for _ in range(len(requests))]

    def update(self):
        # 从缓冲区采样
        states, container_acts, request_acts, rewards, next_states, dones = zip(*self.buffer.sample(batch_size))

        dones = torch.tensor(dones, dtype=torch.float32)
        rewards = torch.tensor(rewards).to(self.device)

        # 拷贝原始档案
        _states = copy.deepcopy(states)
        _next_states = copy.deepcopy(next_states)

        # 转换为张量
        new_states = []
        for state in states:
            temp_state = self.trans_state(state)
            new_states.append(temp_state)
        states = torch.stack(new_states).to(self.device)
        new_states = []
        for next_state in next_states:
            new_states.append(self.trans_state(next_state))
        next_states = torch.stack(new_states).to(self.device)

        # 计算优势
        with torch.no_grad():
            # 应该根据具体的作用在函数中去对state作调整
            values = self.old_policy.critic(states)
            next_values = self.old_policy.critic(next_states)
            rewards = rewards.view(-1, 1)
            next_values = next_values.view(-1, 1)
            dones = dones.view(-1, 1)
            values = values.view(-1, 1)
            advantages = rewards + self.gamma * next_values * (1 - dones) - values

        # 优化策略
        for _ in range(self.K_epochs):
            # 计算新策略概率
            new_container_probs, new_request_logits, new_values = self.policy(states)
            container_acts = torch.tensor(container_acts)

            max_padding_length = self.max_requests * (self.server_number + 1)
            padded_data = []
            for sublist in request_acts:
                padding_needed = max_padding_length - len(sublist)
                padded_sublist = sublist + [0] * padding_needed
                padded_data.append(padded_sublist)
            request_acts = torch.tensor(padded_data)
            # request_acts补0
            request_acts = torch.tensor(request_acts)
            print(new_container_probs.type, container_acts.type)
            # 计算策略损失（带Clip）
            container_ratio = self._compute_prob_ratio(new_container_probs, container_acts)
            routing_ratio = self._compute_prob_ratio(new_request_logits, log_old_probs=request_acts)

            container_loss = self._compute_loss(container_ratio, advantages)
            routing_loss = self._compute_loss(routing_ratio, advantages)

            policy_loss = container_loss + routing_loss

            # 价值损失
            value_loss = nn.MSELoss()(new_values, rewards + self.gamma * next_values)

            # 总损失
            loss = policy_loss + 0.5 * value_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 更新旧策略
        self.old_policy.load_state_dict(self.policy.state_dict())

    def _compute_prob_ratio(self, log_new_probs, log_old_probs):
        """计算新旧策略概率比（需根据动作类型实现）"""
        # 容器部署部分（伯努利分布）
        # 注意：需根据实际采样方式实现
        ratio = torch.exp(log_new_probs - log_old_probs)
        return ratio

    def _compute_loss(self, ratio, advantages):
        batch_size = advantages.size(0)
        ratio = ratio.view(batch_size, -1)
        # assert ratio.size() == advantages.size(), \
        #     'ratio size is {},but advantage size {}'.format(ratio.size(), advantages.size())

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        return policy_loss


if __name__ == "__main__":
    # 配置参数
    config = {
        'state_dim': 107,  # 根据实际状态编码调整
        'num_servers': 3,
        'num_containers': 4,
        'max_requests': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'gamma': 0.99,
        'eps_clip': 0.2,
        'K_epochs': 4,
        'lr': 3e-4,
        'buffer_capacity': 10000
    }

    # 初始化
    agent = PPOAgent(config)
    env = EdgeDeploymentEnv('data')  # 用户实现的环境
    # 训练参数
    max_episodes = 1000
    batch_size = 64
    for episode in range(max_episodes):
        state, done = env.reset()
        episode_reward = 0
        while True:
            # 获得当前的请求序列
            current_requests = env.get_current_requests()
            # 生成动作
            container_act, request_act = agent.act(state, )
            action = {'container_deploy': container_act,
                      'request_assign': request_act}
            # 环境交互
            next_state, reward, done, _ = env.step(action)
            # 存储经验
            agent.buffer.push(state, container_act, request_act, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break
        # 每收集足够经验后更新策略
        if len(agent.buffer) >= batch_size:
            agent.update()
        print(f"Episode {episode}, Reward: {episode_reward:.1f}")
