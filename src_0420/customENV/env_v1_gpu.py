# --------------------------------------------------
# 文件名: env_v1
# 创建时间: 2024/4/20 21:22
# 描述: 考虑了4种资源的限制,环境v1的gpu版本，用torchRL实现
# 作者: WangYuanbo
# --------------------------------------------------
# user_request_info是一个二维数组,[timestamp][user_id]['imageID']，表示ts用户u请求的镜像id
# user_request_info是一个二维数组,[timestamp][user_id]['cpu']
# server_info是一个一维数组 [server_id]['cpu']表示服务器拥有的cpu数量
# server_info是一个一维数组 [server_id]['storage']表示服务器拥有的存储空间
# container_info是一个一维数组 [container_id]['storage']表示容器的存储大小
# container_info是一个一维数组 [container_id]['pulling']表示容器的拉取延迟
from collections import namedtuple
from copy import deepcopy

import gym
import numpy as np
import torch
import yaml
from gym import spaces
from tools import get_server_info, get_container_info, get_user_request_info

# 定义一个具名元组类型
Config = namedtuple('Config',
                    ['user_number',
                     'server_number',
                     'container_number',
                     'penalty',
                     'end_timestamp',
                     'cloud_delay',
                     'edge_delay',
                     'max_cpu',
                     'max_mem',
                     'max_net_in',
                     'max_net_out',
                     'max_disk'])

with open('customENV/env_config.yaml', 'r', encoding='utf-8') as f:
    config_data = yaml.safe_load(f)
config = Config(**config_data)


# user_request_info_dir = 'data/user_request_info.csv'
# container_info_dir = 'data/container_info.csv'
# server_info_dir = 'data/server_info.csv'


class CustomEnv(gym.Env):
    # 定义动作空间和观察空间
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.name = 'env-v1'
        # ########系统数据######
        # 定义一个非法动作的惩罚
        self.penalty = config.penalty
        # 定义终止时间戳
        self.end_timestamp = config.end_timestamp

        # 其他信息
        self.max_cpu = config.max_cpu
        self.max_disk = config.max_mem
        self.max_storage = config.max_disk
        self.max_net_in = config.max_net_in
        self.max_net_out = config.max_net_out

        self.user_number = config.user_number
        self.server_number = config.server_number
        self.container_number = config.container_number
        self.cloud_delay = config.cloud_delay
        self.edge_delay = config.edge_delay

        # 定义请求信息和服务器信息
        self.user_request_info = torch.tensor(get_user_request_info(self.end_timestamp, self.user_number)).to(
            self.device)
        self.server_info = torch.tensor(get_server_info(self.server_number)).to(self.device)
        self.container_info = torch.tensor(get_container_info(self.container_number)).to(self.device)

        self.container_storage = self.container_info[:, 0].view(-1)  # 获得其第一列
        self.container_pulling_delay = self.container_info[:, 1].view(-1)  # 获得第二列

        self.placing_agents_number = self.server_number
        self.routing_agents_number = self.user_number
        self.agents_number = self.placing_agents_number + self.routing_agents_number

        # 系统状态的输入究竟包含哪些？
        # 1.首先对于【cpu,mem,net-in,net-out】supply来说这些都是离散变量4个维度 * server_number
        # 2.

        # 一、 首先是服务器部署哪些镜像,placing_action
        #     1. 上一时刻的服务部署决策 (只用管imageID,不用管他的存储空间是什么样子)
        #     2. 每个容器的存储空间大小
        # 二、 是用户的请求如何路由的问题
        #     1.每个服务器的[cpu,mem,net-in,net-out]supply
        #     2.每个镜像请求的[cpu,mem,net-in,net-out]demand
        #     3.每个镜像请求的imageID
        '''
        总结：每个时刻的系统状态或者观察包含一下内容
        1.每个服务器的[cpu,mem,net-in,net-out,disk]的supply 【属性】
        2.每个镜像的[disk] demand 【属性】
        3.当前时刻每个请求的[cpu,mem,net-in,net-out,lat] demand 【属性】
        4.上一时刻的部署决策[server_number * user_number]  的矩阵
        这就是Q网络的输入
        '''

        self.obs_space = {
            'server': (self.server_number, 5),  # [cpu,mem,net-in,net-out,disk]
            'image_storage': (self.container_number, 2),  # [store,lat]
            'user_request': (self.user_number, 6),  # [id,cpu,mem,net-in,net-out,lat]
            'last_placing_result': (self.server_number, self.container_number),  # 这是一个0，1矩阵
        }
        self.state_dim = 0
        for key, value in self.obs_space.items():
            self.state_dim += value[0] * value[1]
        # 假设每个智能体都能看到环境的全部信息

        self.state_dims = [self.state_dim] * self.agents_number

        # ##########动作空间##########
        # 定义routing动作空间
        routing_action_space = spaces.MultiDiscrete([self.server_number + 1] * self.user_number)
        # 定义placing动作空间
        placing_action_space = spaces.MultiBinary((self.server_number, self.container_number))
        # 最终动作空间
        self.action_space = spaces.Dict({
            "routing": routing_action_space,
            "placing": placing_action_space
        })

        # 定义routing动作空间的输入维度，每个用户的动作空间是server+1
        routing_action_dim = self.server_number + 1
        self.routing_action_dims = [routing_action_dim] * self.routing_agents_number
        # 定义placing动作空间的输入维度，要把s个容器部署到n个服务器上，每个服务器的动作空间是2**s，
        placing_action_dim = (2 ** self.container_number)
        self.placing_action_dims = [placing_action_dim] * self.placing_agents_number

        # 定义初始状态,reset里面去定义
        self.state = {}
        self.timestamp = 0
        self.last_placing_state = None

    def reset(self):
        """
        获得环境的初始状态
        :param self:
        :return:
        """
        self.timestamp = 0
        # reset返回的状态要与obsSpace对应

        # self.obs_space = {
        #     'server': (self.server_number, resource_category),
        #     'image_storage': (self.container_number, 1),
        #     'user_request': (self.user_number, resource_category),
        #     'last_placing_result': (self.server_number, self.container_number),  # 这是一个0，1矩阵
        # }
        server_state = self.server_info
        container_state = self.container_info
        user_request_state = self.user_request_info[self.timestamp]
        last_placing_state = np.zeros((self.server_number, self.container_number))
        self.state = {
            "server_state": server_state,
            "container_state": container_state,
            "user_request_state": user_request_state,
            'last_placing_state': last_placing_state
        }
        # # # 直接把一整变成tensor
        # # user_request_state_df = pd.read_csv(user_request_info_dir)
        # # user_request_state_df.drop('timestamp', inplace=True, axis=1)
        # # user_request_state_df.columns = [None] * len(user_request_state_df.columns)
        # # user_request_state_data = user_request_state_df.to_numpy()
        # # user_request_state = torch.tensor(user_request_state_data).to(self.device)
        # #
        # # 上一时刻的部署决策
        # last_placing_action = np.zeros((self.server_number, self.container_number))
        # # self.last_placing_state = torch.tensor(last_placing_action).to(self.device)
        # # last_placing_action = self.last_placing_state.view(-1)
        # #
        # # # 取出当前时刻的用户请求
        # # now_request_state = user_request_state[0]
        # #
        # # self.state = torch.concat((now_request_state, container_state, server_state, last_placing_action), dim=0)
        return self.state, False

    def step(self, action):
        # 传入一组动作，根据系统当前的状态，输出下一个观察空间
        # 至于这个动作是如何得到在其他模块实现
        # 要返回5个变量，分别是 state (观察状态), reward, terminated, truncated, info

        # 假设传入的action是一个tensordict类型
        placing_action = action['placing_action']
        routing_action = action['routing_action']

        # 传入的placing_action是一个二维0,1矩阵[server_number,container_number]， tensor
        # 传入的routing_action是一个一维[user_number],每个值表示这个用户请求由哪个服务器完成， tensor

        placing_rewards, is_valid_placing_action = self.cal_placing_rewards(placing_action=placing_action)
        routing_rewards, is_valid_routing_action = self.cal_routing_rewards(
            placing_action=placing_action,
            is_valid_placing_action=is_valid_placing_action,
            routing_action=routing_action)

        # 更新系统状态
        now_placing_action = deepcopy(placing_action)
        state = self.state
        # 更新请求状态
        state['user_request_state'] = self.user_request_info[self.timestamp]

        for server_id, container_list in enumerate(now_placing_action):
            if not is_valid_placing_action[server_id]:
                now_placing_action[server_id] = [0] * len(container_list)
        # 更新placing状态
        state['last_placing_state'] = now_placing_action

        self.state = state
        info = placing_rewards + routing_rewards
        info = -np.array(info)
        rewards = sum(info)
        done = False
        self.timestamp += 1
        if self.timestamp == self.end_timestamp:
            done = True
        return self.state, rewards, done, info

    def cal_placing_rewards(self, placing_action):
        """

        :param placing_action: 一个二维张量，server_number* container_number
        :return: placing_rewards, is_valid_placing_action，【server_number】张量
        """
        server_storage_demand = placing_action.clone()

        server_storage_demand *= self.container_storage

        server_storage_demand = torch.sum(server_storage_demand, dim=1)

        is_valid_placing_action = server_storage_demand <= self.server_info[:, 4]  # 取出第五列比较

        # 对placing_action 和 last_placing_state进行比较，得到一个布尔值张量，其中1的位置对应该增加延迟
        delay_mask = (placing_action == 1) & (self.state['last_placing_state'] == 0)

        # 用delay_mask筛选出container_info中的延迟，相乘会将不需要延迟的部分变为0
        delay_values = delay_mask.float() * self.container_pulling_delay

        # 计算出每个server的总延迟
        placing_rewards = delay_values.sum(dim=1)

        # 假设你已经有了 is_valid_placing_action，placing_rewards，以及 self.penalty
        placing_rewards = torch.where(is_valid_placing_action, placing_rewards,
                                      torch.full_like(is_valid_placing_action, self.penalty))

        return placing_rewards, is_valid_placing_action

    def cal_routing_rewards(self, placing_action, is_valid_placing_action, routing_action):
        """
        计算一个step的routing奖励
        :param placing_action: server_number * container_number张量
        :param is_valid_placing_action:  server_number 张量
        :param routing_action: routing_action是一个一维[user_number],每个值表示这个用户请求由哪个服务器完成
        """

        routing_rewards = [0] * self.user_number

        # 首先是要检查每一个routing_action是不是合法的，然后根据
        is_valid_routing_action = [0] * self.user_number
        # 0 表示待定，-1表示非法，1表示路由到云且合法，2表示路由到边且合法
        # 如果路由到云或者路由到非法server，直接是false

        for user_id, server_id in enumerate(routing_action):
            if server_id == self.server_number:  # 路由到云
                is_valid_routing_action[user_id] = 1
            elif not is_valid_placing_action[server_id]:
                is_valid_routing_action[user_id] = -1
            else:  # 如果路由到边，需要边上部署了容器
                image_id = int(self.user_request_info[self.timestamp][user_id][0])
                if placing_action[server_id][image_id] == 0:  # 没有部署相应的镜像，直接给-1，如果部署了，留给下一步检测
                    is_valid_routing_action[user_id] = -1

        # 计算每个服务器的resource_demand, 考虑4种资源类型
        resource_demand = np.zeros((self.server_number, 4))
        for user_id, server_id in enumerate(routing_action):
            if is_valid_routing_action[user_id] == 0:  # 待确认的用户路由
                resource_demand[server_id][0] += self.user_request_info[self.timestamp][user_id][1]  # cpu
                resource_demand[server_id][1] += self.user_request_info[self.timestamp][user_id][2]  # mem
                resource_demand[server_id][2] += self.user_request_info[self.timestamp][user_id][3]  # net-in
                resource_demand[server_id][3] += self.user_request_info[self.timestamp][user_id][4]  # net-out

        for user_id, server_id in enumerate(routing_action):
            if is_valid_routing_action[user_id] == 0:
                if (
                        resource_demand[server_id][0] <= self.server_info[server_id][0]  # cpu
                        and resource_demand[server_id][1] <= self.server_info[server_id][1]  # mem
                        and resource_demand[server_id][2] <= self.server_info[server_id][2]  # net-in
                        and resource_demand[server_id][3] <= self.server_info[server_id][3]  # net-out
                ):
                    is_valid_routing_action[user_id] = 2
                else:
                    is_valid_routing_action[user_id] = -1

        for user_id, is_valid in enumerate(is_valid_routing_action):
            if is_valid_routing_action[user_id] == -1:
                routing_rewards[user_id] = self.penalty
            elif is_valid_routing_action[user_id] == 1:  # 路由到云，奖励是云的传输延迟+计算延迟
                routing_rewards[user_id] = self.cloud_delay + self.user_request_info[self.timestamp][user_id][5]  # lat
            else:  # 路由到边，奖励是边的传输延迟+计算延迟*干扰系数
                routing_rewards[user_id] = self.edge_delay + self.user_request_info[self.timestamp][user_id][5]  # lat


        return routing_rewards, is_valid_routing_action

    def state_to_tensor(self):
        state = self.state
        state_list = []
        for key, value in state.items():
            state_list.append(value.reshape(-1))
        return torch.tensor(np.concatenate(state_list), dtype=torch.float32)


if __name__ == '__main__':
    env = CustomEnv('cpu')
    print(env.container_storage)
    # state, done = env.reset()
