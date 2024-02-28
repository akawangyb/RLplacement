# --------------------------------------------------
# 文件名: placing_routing_env
# 创建时间: 2024/2/28 15:06
# 描述: maddpg请求和路由的环境
# 作者: WangYuanbo
# --------------------------------------------------
# user_request_info是一个二维数组,[timestamp][user_id]['imageID']，表示ts用户u请求的镜像id
# user_request_info是一个二维数组,[timestamp][user_id]['cpu']
# server_info是一个一维数组 [server_id]['cpu']表示服务器拥有的cpu数量
# server_info是一个一维数组 [server_id]['storage']表示服务器拥有的存储空间
# container_info是一个一维数组 [container_id]['storage']表示容器的存储大小
# container_info是一个一维数组 [container_id]['pulling']表示容器的拉取延迟
from typing import Tuple

import gym
import numpy as np
import yaml
from gym import spaces
from gym.core import ActType, ObsType

from tools import get_server_info, get_container_info, get_user_request_info


class CustomEnv(gym.Env):
    # 定义动作空间和观察空间
    def __init__(self, server_info, user_request_info, container_info, penalty, config):
        super().__init__()

        # ########系统数据######
        # 定义一个非法动作的惩罚
        self.penalty = penalty
        # 定义终止时间戳
        self.end_timestamp = 10

        # 定义请求信息和服务器信息
        self.user_request_info = user_request_info
        self.server_info = server_info
        self.container_info = container_info

        # 其他信息
        self.max_cpu = config['max_cpu']
        self.max_storage = config['max_storage']
        self.user_number = config['user_number']
        self.server_number = config['server_number']
        self.container_number = config['container_number']
        self.cloud_delay = config['cloud_delay']
        self.edge_delay = config['edge_delay']

        self.placing_agents_number = self.server_number
        self.routing_agents_number = self.user_number
        self.agents_number = self.placing_agents_number + self.routing_agents_number

        # ######routing_agent观察空间#######
        # 1. 每个服务器的cpu资源（0，n）上的连续值
        server_cpu_space = spaces.Box(low=0, high=self.max_cpu, shape=(self.server_number,),
                                      dtype=np.float32)

        # 2.当前用户请求的镜像id
        user_request_imageID_space = spaces.MultiDiscrete([self.container_number] * self.user_number)

        # 3.定义用户请求占用的cpu大小
        user_request_cpu_space = spaces.Box(low=0, high=self.max_cpu, shape=(self.user_number,),
                                            dtype=np.float32)

        # 4.上一时刻的镜像部署情况，如果能看见这个信息，优先路由到上一时刻已经部署的服务器上。
        last_placing_action_space = spaces.MultiBinary((self.server_number, self.container_number))

        # self.routing_observation_space = spaces.Dict({
        #     'server_cpu': server_cpu_space,
        #     'user_request_imageID': user_request_imageID_space,
        #     'user_request_cpu': user_request_cpu_space,
        #     'last_placing_action': last_placing_action_space
        # })

        # ######placing_agent观察空间#######
        # 1.上一个时刻的placing结果,二维0，1矩阵,上面已经定义。
        # 5.每个服务器的存储空间大小
        server_storage_space = spaces.Box(low=0, high=self.max_storage, shape=(self.server_number,),
                                          dtype=np.float32)
        # 6.每个容器的镜像大小
        container_storage_space = spaces.Box(low=0, high=self.max_storage, shape=(self.container_number,),
                                             dtype=np.float32)

        # self.placing_observation_space = spaces.Dict({
        #     'last_placing_action': last_placing_action_space,
        #     'server_storage': server_storage_space,
        #     'container_storage': container_storage_space
        # })

        # 定义最终的观察空间
        # 在这个例子里面假设一共有10个服务，3个边缘服务器，10个用户
        self.observation_space = spaces.Dict({
            'server_storage': server_storage_space,
            'container_storage': container_storage_space,
            'last_placing_action': last_placing_action_space,
            'server_cpu': server_cpu_space,
            'user_request_imageID': user_request_imageID_space,
            'user_request_cpu': user_request_cpu_space,
        })
        self.state_dim = 0
        for item in self.observation_space:
            dim = 1
            for shape in self.observation_space[item].shape:
                dim *= shape
            self.state_dim += dim
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

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # 传入一个动作，根据系统当前的状态，输出下一个观察空间
        # 至于这个动作是什么在其他模块实现
        # 要返回5个变量，分别是 state (观察状态), reward, terminated, truncated, info
        # 获得当前的环境的routing和placing状态
        # 获得当前传入的各个员工的动作
        placing_action = action[:self.placing_agents_number]

        routing_action = action[-self.routing_agents_number:]
        state = self.state
        reward = [0] * self.agents_number
        terminated = False
        truncated = False
        info = {}
        # 根据这个动作计算奖励值，
        # 把传入action分成2部分，placing_action,routing_action
        # 不合法的placing，给一个惩罚，并且不改变系统状态，
        # 如果routing到了一个非法的placing，给一个惩罚。
        # 如果routing的上一个系统状态是合法的

        # 先把上一个时隙的placing动作找出来
        last_placing_action = state['last_placing_action'].reshape(self.server_number, self.container_number)
        # 解决placing的约束
        placing_rewards = [0] * self.server_number
        placing_matrix = [None] * self.server_number
        # 找出合法的服务器镜像部署动作
        valid_server_list = [True] * self.server_number
        for server_id, server_action in enumerate(placing_action):
            # 获得具体的部署动作
            server_action = np.argmax(server_action)
            # server_action是一个十进制数，要把他转换成二进制list
            server_action = np.binary_repr(server_action, width=self.container_number)
            placing_matrix[server_id] = server_action
            server_storage_demand = sum([
                int(is_placed) * self.container_info[container_id]['storage']
                for container_id, is_placed in enumerate(server_action)
            ])

            if self.server_info[server_id]['storage'] <= server_storage_demand:
                valid_server_list[server_id] = False
                # 也就是负责服务器i部署的智能体获得一个惩罚
                placing_rewards[server_id] += self.penalty
            else:  # 否则获得奖励是前后的容器拉取延迟
                for container_id, is_placed in enumerate(last_placing_action[server_id]):
                    placing_rewards[server_id] += is_placed * self.container_info[container_id]['pulling']
        now_placing_action = []
        for server_id in range(self.server_number):
            if valid_server_list[server_id]:
                now_placing_action.append(np.array(list(last_placing_action[server_id]), dtype=int))
            else:
                now_placing_action.append([0] * self.container_number)

        # 再解决routing的奖励
        routing_rewards = [0] * self.user_number
        # 约束一，必须部署相应的镜像
        for user, user_action in enumerate(routing_action):
            imageID = state['user_request_imageID'][user]
            # 这里要检查一下
            server_index = np.argmax(user_action)
            if server_index == self.server_number:
                reward[user] += self.cloud_delay  # 路由到云服务器的奖励为0
            # 路由到错误部署的服务器，获得一个惩罚
            elif placing_matrix[server_index][imageID] == 0:
                reward[user] += self.penalty
            elif not valid_server_list[server_index]:
                reward[user] += self.penalty
            else:
                reward[user] += self.edge_delay  # 路由到边缘服务器的奖励，是其执行时间

        # 约束二，cpu计算量不能超
        cpu_demand = [0] * self.server_number
        # 先找出哪个服务器超过了
        for user, user_action in enumerate(routing_action):
            server_index = np.argmax(user_action)
            if server_index == self.server_number:
                continue
            else:
                cpu_demand[server_index] += state['user_request_cpu'][user]

        invalid_server_list = []
        for server_index, cpu in enumerate(cpu_demand):
            if cpu > state['server_cpu'][server_index]:
                invalid_server_list.append(server_index)

        # 超过cpu的智能体获得惩罚
        for invalid_server in invalid_server_list:
            for user, user_action in enumerate(action):
                server_index = np.argmax(user_action)
                if server_index == invalid_server:
                    reward[user] += self.penalty

        # 更新系统时间戳状态
        self.timestamp += 1
        ts = self.timestamp
        # 判断是否达到了截断条件和终止条件?
        if ts == self.end_timestamp:
            terminated = True
        else:  # 更新系统状态
            # 用户请求cpu
            state['user_request_cpu'] = np.array(
                [self.user_request_info[ts][user]['cpu'] for user in range(self.user_number)])
            # 用户请求镜像id
            state['user_request_imageID'] = np.array([self.user_request_info[ts][user]['imageID'] for user in
                                                      range(self.user_number)])
            state['last_placing_action'] = np.array(now_placing_action).reshape(-1)
        done = terminated or truncated
        self.state = state
        return [self.state] * self.agents_number, reward, [done] * self.agents_number, info

    def reset(self):
        # self.timestamp = 0
        # reset返回的状态要与obsSpace对应
        initial_cpu = []
        initial_imageID = []

        for user_info in self.user_request_info[0]:
            initial_cpu.append(user_info['cpu'])
            initial_imageID.append(user_info['imageID'])
        initial_cpu = np.array(initial_cpu)
        initial_imageID = np.array(initial_imageID)
        initial_placing = np.zeros((self.server_number, self.container_number,))

        container_storage = []
        for a_container_info in self.container_info:
            container_storage.append(a_container_info['storage'])
        container_storage = np.array(container_storage)

        # self.observation_space = spaces.Dict({
        #     'server_storage': server_storage_space,
        #     'container_storage': container_storage_space,
        #     'last_placing_action': last_placing_action_space,
        #     'server_cpu': server_cpu_space,
        #     'user_request_imageID': user_request_imageID_space,
        #     'user_request_cpu': user_request_cpu_space,
        # })
        self.state = {
            'server_storage': np.full((self.server_number,), self.max_storage, dtype=np.float32),
            'container_storage': container_storage,
            'last_placing_action': initial_placing,
            'server_cpu': np.full((self.server_number,), self.max_cpu, dtype=np.float32),
            'user_request_cpu': initial_cpu,
            'user_request_imageID': initial_imageID,
        }

        self.timestamp = 0
        return [self.state] * self.agents_number, [False] * self.agents_number


if __name__ == '__main__':
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    penalty = 100
    server_info = get_server_info()
    container_info = get_container_info()
    user_request_info = get_user_request_info(config['user_number'])

    config['max_cpu'] = 5
    config['max_storage'] = 100000
    config['cloud_delay'] = 10
    config['edge_delay'] = 5
    env = CustomEnv(server_info=server_info, container_info=container_info, user_request_info=user_request_info,
                    penalty=penalty, config=config)
    states, dones = env.reset()
    print(len(states[0]['last_placing_action']))
    print(env.state_dim)
    # print(env.action_space)
    # print(env.routing_state_dims)
    # print(env.routing_action_dims)
    # print(env.placing_state_dims)
    # print(env.placing_action_dims)
    # states前几项是placing智能体的观察空间，后几项是routing智能体的观察空间
