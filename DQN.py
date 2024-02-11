# --------------------------------------------------
# 文件名: DQN
# 创建时间: 2024/2/4 19:13
# 描述: 创建一个简单的dqn试一下，基于简单dqn，实现干扰感知的容器的部署
# 先假设只考虑2种资源，一个是存储，一个是cpu
# 作者: WangYuanbo
# --------------------------------------------------
from typing import Tuple

import gym
import numpy as np
from gym import spaces
from gym.core import ActType, ObsType


# 边缘服务器的信息,这个是全局的信息，与动作无关

# 首先要定义观察空间和动作空间，
# 观察空间一共包含3个信息
# 1.上一时刻的容器部署信息
# 2.上一时刻的请求路由信息
# 3.当前时刻的服务请求的信息
# 这里用字典表示三项不同的内容

# 假设所有的服务器都是同构的，考虑cpu和disk两种资源
# 假设有5个服务器，每一个16个cpu，1000GB存储。
# 服务的数量是固定的100个，
# 假设每个时刻的请求数量也是固定的500个。

# 事实上，每个服务器有多少资源空间，每个容器需要消耗多少空间是固定的。

# 强化学习的大致流程是 当前状态->选择动作(e-greedy)->计算奖励&下一个状态—>回到第一步
# 在我的这个例子里面，下一步状态与
class CustomEnv(gym.Env):
    # 定义动作空间和观察空间
    def __init__(self, config):
        super().__init__()
        # 定义一个非法动作的惩罚
        self.penalty = config['penalty']
        # 定义终止时间戳
        self.end_timestamp = config['end_timestamp']
        # 定义每个容器镜像的拉取时延
        self.images_pulling_delay = config['images_pulling_delay']

        # 定义环境本身的配置，服务器数量，容器数量等。
        self.server_number = config['server_number']
        self.container_number = config['container_number']
        self.request_number = config['request_number']
        # 除此以外，每个容器需要多少存储空间，也是环境信息
        # 假设这个信息，可以从配置文件中读取
        self.container_storage = config['container_storage']
        self.container_cpu = config['container_cpu']

        # 定义每个用户的请求需要多少cpu计算资源，也是环境信息
        self.request_resource = config['request_resource']

        # 定义每个时隙的用户请求,
        # 每个时隙的用户请求是一个一维数组
        # config传入的全体请求就是一个二维数组
        self.user_request_per_ts = config['user_request_per_ts']

        # ######定义观察空间#######
        # 1. 定义上一时刻的容器部署信息,x_{n,s}={0,1}表示服务器n上是否部署容器s
        # 此处用一个二维矩阵表示
        self.last_container_place_space = spaces.MultiBinary((self.server_number, self.container_number))

        # 2. 定义上一时刻的请求路由信息，y_{r_s,n}={0,1}关于服务s的请求是否由边缘服务器n完成。
        # 此处也用一个二维矩阵表示
        self.last_request_routing_space = spaces.MultiBinary((self.request_number, self.server_number + 1))

        # 3. 定义此时的用户服务请求
        # 假设请求数量是一定的，例如请求有500个，服务100个。
        # 对于每一个用户请求的路由，有服务器数量+1种可能
        single_user_request_space = spaces.Discrete(self.server_number + 1)
        self.user_request_space = spaces.Tuple([single_user_request_space] * self.request_number)

        # 4.定义当前时间戳
        self.timestamp_space = spaces.Discrete(self.end_timestamp + 1)

        # 定义最终的观察空间
        self.observation_space = spaces.Dict({
            'last_container_placement': self.last_container_place_space,
            'last_request_routing': self.last_request_routing_space,
            'user_request': self.user_request_space,
            'timestamp': self.timestamp_space,
        })
        # ##########动作空间##########
        # 定义动作空间，两个动作
        # 1. 对于每个服务器，部署哪些容器,x_{n,s}={0,1}表示服务器n上是否部署容器s？
        # 似乎就是上面的东西？
        self.now_container_place_space = spaces.MultiBinary((self.server_number, self.container_number))
        # 2. 对于每个请求，路由到哪个服务器？
        self.now_request_routing_space = spaces.MultiBinary((self.request_number, self.server_number))
        self.action_space = spaces.Dict({
            'now_container_place': self.now_container_place_space,
            'now_request_routing': self.now_request_routing_space
        })
        # 定义初始状态
        # 初始状态所有都是空的，
        # # 时间戳决定了到达的请求的集合
        # self.state = {
        #     'last_container_placement': np.zeros((config['server_number'], config['container_number'])),
        #     'last_request_routing': np.zeros((config['request_number'], config['server_number'] + 1)),
        #     'user_request': np.zeros(config['request_number'])
        # }

    # 执行一个动作，返回新的状态、奖励、以及是否完成
    # 至于这个动作是如何选择的，这里不需要管
    # 对于非法动作的处理：添加一个很大的负值作为惩罚
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # 传入一个动作，根据系统当前的状态，输出下一个观察空间
        # 至于这个动作如何选择在其他模块实现

        # 要返回5个变量，分别是 state (观察状态), reward, terminated, truncated, info
        state = self.state
        reward = 0
        terminated = False
        truncated = False
        info = {}
        # 根据这个动作计算奖励值，
        # 如果动作是合法的，获得奖励，并更新状态
        # 如果动作是非法的，要更新时间戳状态，不更新上一个部署的状态
        if not self.isValid(action):
            reward += self.penalty
        else:
            reward += self.getReward(action, state)
            # 更新上一个部署系统状态
            state["last_container_placement"] = action["now_container_place"]
            state["last_request_routing"] = action["now_request_routing"]
        # 更新系统时间戳状态
        state['timestamp'] += 1

        # 根据下一个时间戳更新下一个时间戳的系统状态
        ts = state['timestamp']
        state['user_request'] = self.user_request_per_ts[ts]

        # 判断是否达到了截断条件和终止条件?
        if ts > self.end_timestamp:
            terminated = True

        return self.state, reward, terminated, truncated, info

    # 初始化环境状态
    def reset(self):
        # self.timestamp = 0
        # reset返回的状态要与obsSpace对应
        self.state = {
            'timestamp': 1,
            'last_container_placement': np.zeros((self.server_number, self.container_number)),
            'last_request_routing': np.zeros((self.request_number, self.server_number + 1)),
            'user_request': np.zeros(self.request_number)
        }
        return self.state

    # 检查动作是不是合法的
    def isValid(self, action: ActType) -> bool:
        x = action['now_container_place']
        y = action['now_request_routing']
        # 约束一
        # 检查是不是所有的用户请求都有一个服务器完成
        for u in range(self.request_number):
            count_server = 0
            for n in range(self.server_number + 1):
                count_server += y[u][n]
            if count_server != 1:
                return False
        # 约束二
        # 检查请求路由的边缘服务器是否部署了对应的服务
        for u in range(self.request_number):
            for n in range(self.server_number):
                for s in range(self.container_number):
                    if y[u][n] > x[n][s]:
                        return False
        # 约束三
        # 对于服务部署，检查容器的的磁盘空间是不是满足的
        for n in range(self.server_number):
            # 计算服务器n上的存储空间
            n_storage = 0
            for s in range(self.container_number):
                n_storage += x[n][s] * self.container_storage[s]
            if n_storage > self.container_storage[n]:
                return False
        # 约束四
        # 对于请求路由，首先检查服务器的cpu资源是不是足够的
        for n in range(self.server_number):
            n_cpu = 0
            for u in range(self.request_number):
                n_cpu += y[u][n] * self.request_resource[u]
            if n_cpu > self.container_cpu[n]:
                return False
        # 需要考虑请求覆盖的问题
        # 这个约束在建模里面没有

    # 对于一个合法的动作计算他获得的奖励
    # 初始版本，只计算镜像拉取带来的延迟
    def getReward(self, action: ActType, state: ObsType) -> int:
        # 如果当前时间戳是1，直接计算拉取延迟
        x = action['now_container_place']
        b = self.images_pulling_delay
        total_delay = 0
        delay_pulling = 0
        if state.timestamp == 0:
            for n in range(self.server_number):
                for s in range(self.container_number):
                    delay_pulling += x[n][s] * b[s]
        total_delay += delay_pulling
        return -total_delay
