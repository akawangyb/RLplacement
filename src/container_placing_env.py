# --------------------------------------------------
# 文件名: container_placing_env
# 创建时间: 2024/2/24 16:29
# 描述: 描述容器部署的环境
# 作者: WangYuanbo
# --------------------------------------------------
# 定义好状态空间和动作空间
from typing import Tuple

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.core import ActType, ObsType


class CustomEnv(gym.Env):
    # 定义动作空间和观察空间
    def __init__(self, server_storage_size, container_info, penalty):
        super().__init__()

        # ########系统数据######
        # 定义一个非法动作的惩罚
        self.penalty = penalty
        # 定义终止时间戳
        self.end_timestamp = len(container_info)

        # 定义每个容器镜像信息,包括id,大小,拉取时延
        self.container_info = container_info
        self.server_storage_size = server_storage_size
        # ######定义观察空间#######
        # 1. 当前的服务器的存储空间（0，n）上的连续值
        # self.server_storage_space = spaces.Box(low=0, high=self.server_storage_size, shape=(1,), dtype=np.float32)

        # 2. 定义此时容器镜像的存储大小（0，n）上的连续值
        # self.container_storage_space = spaces.Box(low=0, high=self.container_storage_size, shape=(1,), dtype=np.float32)

        # 定义最终的观察空间
        self.observation_space = spaces.Box(low=0, high=self.server_storage_size, shape=(2,), dtype=np.float32)

        # ##########动作空间##########
        # 定义动作空间，两个动作
        # 1. 一个动作，0，1，是否部署容器s到服务器n上
        self.action_space = spaces.Discrete(2)

        # 定义动作空间的输入维度
        self.action_dim = 2
        # 定义状态空间的输入维度
        self.state_dim = 2
        # 定义初始状态
        # 初始状态所有都是空的，
        # # 时间戳决定了到达的请求的集合
        self.state = np.zeros(2)
        self.timestamp = 0

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # 传入一个动作，根据系统当前的状态，输出下一个观察空间
        # 至于这个动作是什么在其他模块实现
        # 要返回5个变量，分别是 state (观察状态), reward, terminated, truncated, info
        state = self.state
        reward = 0
        terminated = False
        truncated = False
        info = {}
        # 根据这个动作计算奖励值，
        # 如果动作是合法的，获得奖励，并更新状态
        # 如果动作是非法的，要更新时间戳状态，不更新上一个部署的状态
        if not self.isValid(action, state):
            reward += self.penalty
            print("not valid action")
        else:
            reward += action
            # 更新服务器存储状态
            state[0] -= action * state[1]
        # 更新系统时间戳状态
        self.timestamp += 1
        ts = self.timestamp
        # 判断是否达到了截断条件和终止条件?
        if ts == self.end_timestamp:
            terminated = True
        else:  # 更新用户的请求信息
            state[1] = self.container_info[ts]
        done = terminated or truncated
        self.state = state
        return self.state, reward, done, info

    def isValid(self, action, state):
        if action == 1 and state[1] > state[0]:
            return False
        return True

    def reset(self):
        # self.timestamp = 0
        # reset返回的状态要与obsSpace对应
        self.state = np.array([self.server_storage_size, self.container_info[0]])
        self.timestamp = 0
        return self.state


if __name__ == '__main__':
    server_storage_size = 100000
    container_info = []
    file_path = '../data/container_info.csv'
    key = 'container_id'
    data_frame = pd.read_csv(file_path)
    data_frame.set_index(key, inplace=True)
    data_dict = data_frame.to_dict('index')
    print(data_dict)
    for key in data_dict:
        container_info.append(data_dict[key]['container_size'])
    print(container_info)
    # config[file_name.replace('.csv', '')] = data_dict

    penalty = -10
    env = CustomEnv(server_storage_size=server_storage_size, container_info=container_info, penalty=penalty)
    state = env.reset()
    print(state)
