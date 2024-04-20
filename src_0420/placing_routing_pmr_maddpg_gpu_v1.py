# --------------------------------------------------
# 文件名: placing_routing_pmr_maddpg_v1
# 创建时间: 2024/3/3 21:55
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
# 相对于v0优化一下代码
# 首先搞清楚ddpg的流程
# 有两个网络，一个actor（动作）网络，一个critic（评价）网络
# 动作网络的给出当前状态下应该执行的动作
# 评价网络评估这个动作的价值，评价网络的输入是（当前状态，当前动作）
# 评价网络的更新目标是最小化贝尔曼误差
# 试图让我们的Critic预测的动作值函数尽可能的接近感知到的长期回报。
# y_i = r_i + γ * Q(s_{i+1}, a_{i+1})
# r_i代表了在当前状态 s_i 执行动作 a_i 后得到的奖励，
# γ 是折扣因子，Q(s_{i+1}, a_{i+1}) 表示执行下一步动作 a_{i+1}
# 在状态 s_{i+1} 下由目标网络（Target Network）给出的预测动作值。

#  如何更新动作网络了?动作的网络的目的是生成Q值更大的动作,
# 因此其更新目标就是最小化评价网络Q值的相反数,
# loss= -torch.mean(self.critic(state,self.actor(state)))
import argparse
import sys
import time
from datetime import datetime

import yaml
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter

from customENV.placing_routing_env_gpu_v1 import CustomEnv
from memory.buffer import MAPrioritizedReplayBuffer
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
        # print(action.shape, action)
        # return action.detach().cpu().numpy()[0]
        return action.view(-1)

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
        # routing智能体公用同一个状态
        # placing智能体公用同一个状态
        # routing系统状态是Dict类型，只需要把他处理即可
        state = torch.tensor(state, dtype=torch.float).to(self.device).view(1, -1)
        actions = [
            agent.take_action(state, explore) for agent in self.agents
        ]
        # actions_tensor = torch.tensor(actions).to(self.device)
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
        critic_loss.backward(retain_graph=True)
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
            agent.actor.torch.load(name).to(self.device)


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
env = CustomEnv()

# 获得脚本名字
filename = os.path.basename(sys.argv[0])  # 获取脚本文件名
script_name = os.path.splitext(filename)[0]  # 去除.py后缀
# 日志输出路径
father_log_directory = 'log'
if not os.path.exists(father_log_directory):
    os.makedirs(father_log_directory)
current_time = datetime.now()
formatted_time = current_time.strftime('%Y%m%d-%H_%M_%S')
log_file_name = script_name + '_' + env.name + '-' + formatted_time

log_path = os.path.join(father_log_directory, log_file_name)
# 规范化文件路径
log_dir = os.path.normpath(log_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'output_info.log')
f_log = open(log_path, 'w', encoding='utf-8')
writer = SummaryWriter(log_dir)

with open('train_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

config = edict(config)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

replay_buffer = MAPrioritizedReplayBuffer(
    state_size=env.state_dim,
    placing_action_size=env.placing_action_dims[0],
    routing_action_size=env.routing_action_dims[0],
    placing_agent_number=env.placing_agents_number,
    routing_agent_number=env.routing_agents_number,
    device=device,
    buffer_size=config.buffer_size,
)

routing_action_dims = env.routing_action_dims
placing_action_dims = env.placing_action_dims
action_dims = placing_action_dims + routing_action_dims
state_dims = env.state_dims
critic_input_dim = env.state_dim + sum(action_dims)

maddpg = MADDPG(env=env, action_dims=action_dims, state_dims=state_dims,
                critic_input_dim=critic_input_dim,
                hidden_dim=config.hidden_dim, actor_lr=config.actor_lr, critic_lr=config.critic_lr,
                tau=config.tau, gamma=config.gamma, device=device, log_dir=log_dir)


def evaluate(para_env, maddpg, n_episode=10):
    # 对学习的策略进行评估,此时不会进行探索
    env = para_env
    returns = np.zeros(env.agents_number)
    for _ in range(n_episode):
        state, done = env.reset()
        while not done:
            actions = maddpg.take_action(state, explore=False)
            state, rewards, done, info = env.step(actions)  # rewards输出的是所有agent的和，info输出的是每个agent的reward
            rewards = np.array(info)
            returns += rewards * 1.0
    returns = returns / n_episode
    returns = returns.tolist()
    # 输出每个agent的reward，以及全体agent的和
    f_log.write("Episode: {}, total reward: {}\n".format(i_episode + 1, sum(returns)))
    f_log.write("each agent reward: {}\n".format(returns))
    print(f"Episode {i_episode + 1} : {returns}")
    print(f"sum of returns: {sum(returns)}")
    return returns


total_step = 0
best_reward = {
    'value': -np.inf,
    'list': [],
    'episode': -1,
}
last_time = time.time()
for i_episode in range(config.num_episodes):
    state, done = env.reset()
    while not done:
        actions = maddpg.take_action(state, explore=True)
        # for action in actions:
        # print(type(actions), actions)
        # 这里输出的actions是一个list的np array
        # 主要是这一步，action与环境进行交互
        next_state, rewards, done, _ = env.step(actions)
        placing_actions = actions[:env.placing_agents_number]
        routing_actions = actions[-env.routing_agents_number:]
        replay_buffer.add((state, placing_actions, routing_actions, rewards, next_state, done))
        state = next_state
        total_step += 1
        if replay_buffer.real_size >= config.minimal_size and total_step % config.update_interval == 0:
            batch, weights, tree_idxs = replay_buffer.sample(config.batch_size)
            all_agents_td_error = []
            for a_i in range(env.agents_number):
                print(a_i)
                loss, td_error = maddpg.update(batch, a_i)
                all_agents_td_error.append(td_error)
            all_agents_td_error = torch.stack(all_agents_td_error, dim=1)
            all_agents_td_error = torch.sum(all_agents_td_error, dim=1).reshape(-1)
            replay_buffer.update_priorities(tree_idxs, all_agents_td_error.cpu().numpy())
            maddpg.update_all_targets()
    if (i_episode + 1) % 10 == 0:
        # ep_returns是一个np

        ep_returns = evaluate(env, maddpg, n_episode=10)
        now_time = time.time()
        elapsed_time = round(now_time - last_time, 2)
        last_time = now_time
        print(elapsed_time)
        # 保存最优的模型
        if sum(ep_returns) > best_reward['value']:
            best_reward['value'] = sum(ep_returns)
            best_reward['list'] = ep_returns
            best_reward['episode'] = i_episode + 1
            maddpg.save()

print(best_reward['episode'], best_reward['value'], best_reward['list'])

# # 训练结束进行测试
# # 初始化13个actor网络
# maddpg.load_model()
# state, done = env.reset()
# while not done:
#     actions = maddpg.take_action(state, explore=False)
#     print(actions)
#     next_state, rewards, done, _ = env.step(actions)
#     print(rewards)
#     state = next_state
