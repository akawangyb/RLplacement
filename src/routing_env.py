# --------------------------------------------------
# 文件名: routing_env
# 创建时间: 2024/2/26 10:22
# 描述: 完成用户路由请求的环境
# 作者: WangYuanbo
# --------------------------------------------------
# user_request_info是一个二维数组,[timestamp][user_id]['imageID']，表示ts用户u请求的镜像id
# user_request_info是一个二维数组,[timestamp][user_id]['cpu']
# server_info是一个一维数组 [server_id]['cpu']表示服务器拥有的cpu数量
# server_info是一个一维数组 [server_id]['storage']表示服务器拥有的存储空间
# container_info是一个一维数组 [container_id]['storage']表示容器的存储大小
# container_info是一个一维数组 [container_id]['pulling']表示容器的拉取延迟
# 定义好状态空间和动作空间
from typing import Tuple

import gym
import numpy as np
import yaml
from gym import spaces
from gym.core import ActType, ObsType

from tools import get_server_info, get_container_info, get_user_request_info, get_placing_info


class CustomEnv(gym.Env):
    # 定义动作空间和观察空间
    def __init__(self, server_info, user_request_info, container_info, placing_info, penalty, config):
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
        self.placing_info = placing_info

        # 其他信息
        self.max_cpu = config['max_cpu']
        self.user_number = config['user_number']
        self.server_number = config['server_number']
        self.container_number = config['container_number']

        self.agents = self.user_number

        # # 定义每个容器镜像信息,包括id,大小,拉取时延
        # self.container_info = container_info
        # self.server_storage_size = server_storage_size
        # ######定义观察空间#######
        # 1. 每个服务器的cpu资源（0，n）上的连续值
        self.server_cpu_space = spaces.Box(low=0, high=self.max_cpu, shape=(self.server_number,),
                                           dtype=np.float32)

        # 2.当前用户请求的镜像id
        self.user_request_imageID_space = spaces.MultiDiscrete([self.container_number] * self.user_number)

        # 3.定义用户请求占用的cpu大小
        self.user_request_cpu_space = spaces.Box(low=0, high=self.max_cpu, shape=(self.user_number,),
                                                 dtype=np.float32)

        # 定义最终的观察空间
        # 在这个例子里面假设一共有100个服务，每个服务器随机部署了50个服务。
        self.observation_space = spaces.Dict({
            'server_cpu': self.server_cpu_space,
            'user_request_cpu': self.user_request_cpu_space,
            'user_request_imageID': self.user_request_imageID_space
        })

        # ##########动作空间##########
        # 定义动作空间
        self.action_space = spaces.MultiDiscrete([self.server_number + 1] * self.user_number)

        # 定义状态空间的输入维度
        state_dim = self.server_cpu_space.shape[0] + self.user_request_imageID_space.shape[
            0] + self.container_number * self.user_number
        self.state_dims = [state_dim] * self.user_number

        # 定义动作空间的输入维度
        self.action_dims = [self.server_number + 1] * self.user_number

        # 定义初始状态
        # 初始状态所有都是空的，
        # # 时间戳决定了到达的请求的集合
        print(self.state_dims[0])
        self.state = {}
        self.timestamp = 0

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # 传入一个动作，根据系统当前的状态，输出下一个观察空间
        # 至于这个动作是什么在其他模块实现
        # 要返回5个变量，分别是 state (观察状态), reward, terminated, truncated, info
        state = self.state
        reward = [0] * self.agents
        terminated = False
        truncated = False
        info = {}
        # 根据这个动作计算奖励值，
        # 如果动作是合法的，获得奖励，并更新状态
        # 如果动作是非法的，要更新时间戳状态，不更新上一个部署的状态
        # 这里的动作的奖励重新设置
        # 假如路由到一个服务器没有部署镜像，惩罚是-1
        # 假如路由到云服务器，不奖不罚
        # 假如路由正确，奖励+1
        # 对于cpu资源限制如果不满足就-5

        # action = action.tolist()
        # print(action)
        # 约束一，必须部署相应的镜像
        for user, user_action in enumerate(action):
            # print(state['user_request_imageID'][user])
            imageID = state['user_request_imageID'][user]
            server_index = np.argmax(user_action)
            if server_index == self.server_number:
                continue
            # print(user_action, imageID)
            if self.placing_info[server_index][imageID] == 0:
                reward[user] -= 1
            else:
                reward[user] += 1
                state['server_cpu'][server_index] -= state['user_request_cpu'][server_index]

        # 约束二，cpu计算量不能超
        cpu_demand = [0] * self.server_number
        # 先找出哪个服务器超过了
        for user, user_action in enumerate(action):
            server_index = np.argmax(user_action)
            if server_index == self.server_number:
                continue
            else:
                cpu_demand[server_index] += state['user_request_cpu'][user]

        invalid_server_list = []
        for server_index, cpu in enumerate(cpu_demand):
            if cpu > state['server_cpu'][server_index]:
                invalid_server_list.append(server_index)

        # 把刚才减掉的加回来
        for invalid_server in invalid_server_list:
            for user, user_action in enumerate(action):
                server_index = np.argmax(user_action)
                if server_index == invalid_server:
                    reward[user] -= 1
                    state['server_cpu'][server_index] += state['user_request_cpu'][server_index]

        # 更新系统时间戳状态
        self.timestamp += 1
        ts = self.timestamp
        # 判断是否达到了截断条件和终止条件?
        if ts == self.end_timestamp:
            terminated = True
        else:  # 更新用户的请求信息
            state['user_request_cpu'] = np.array(
                [self.user_request_info[ts][user]['cpu'] for user in range(self.user_number)])
            state['user_request_imageID'] = np.array([self.user_request_info[ts][user]['imageID'] for user in
                                                      range(self.user_number)])
        done = terminated or truncated
        self.state = state
        return [self.state] * self.agents, reward, [done] * self.agents, info

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
        self.state = {
            'server_cpu': np.full((self.server_number,), self.max_cpu, dtype=np.float32),
            'user_request_cpu': initial_cpu,
            'user_request_imageID': initial_imageID
        }
        self.timestamp = 0
        return [self.state] * self.agents, [False] * self.agents


if __name__ == '__main__':
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    penalty = -10
    server_info = get_server_info()
    container_info = get_container_info()
    user_request_info = get_user_request_info(config['user_number'])
    placing_info = get_placing_info()

    config['max_cpu'] = 5
    env = CustomEnv(server_info=server_info, container_info=container_info, user_request_info=user_request_info,
                    placing_info=placing_info, penalty=penalty, config=config)
    state = env.reset()
    print(env.observation_space)
    print(env.action_space)
    print(env.action_dims)
    print(env.state_dims)
    # print(state)
