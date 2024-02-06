# --------------------------------------------------
# 文件名: DQN
# 创建时间: 2024/2/4 19:13
# 描述: 创建一个简单的dqn试一下
# 作者: WangYuanbo
# --------------------------------------------------

import gym
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
    def __init__(self, config):
        super().__init__()

        # # Define action and observation space
        # self.action_space = spaces.Discrete(2)  # 移动左或右
        # self.observation_space = spaces.Discrete(10)  # 位置，范围为0-9
        # 定义环境本身的配置，服务器数量，容器数量等。
        self.config = config
        # self.container_number

        # 1. 定义上一时刻的容器部署信息,x_{n,s}={0,1}表示服务器n上是否部署容器s
        # 此处用一个二维矩阵表示
        self.last_container_place_space = spaces.MultiBinary((config['server_number'], config['container_number']))

        # 2. 定义上一时刻的请求路由信息，y_{r_s,n}={0,1}关于服务s的请求是否由边缘服务器n完成。
        # 此处也用一个二维矩阵表示
        self.last_request_routing_space = spaces.MultiBinary((config['request_number'], config['server_number']))

        # 3. 定义此时的用户服务请求
        # 假设请求数量是一定的，例如请求有500个，服务100个。
        single_user_request_space = spaces.Discrete(config['service_number'])
        self.user_request_space = spaces.Tuple([single_user_request_space] * config['request_number'])

        # 定义最终的观察空间
        self.observation_space = spaces.Dict({
            'last_container_placement': self.last_container_place_space,
            'last_request_routing': self.last_request_routing_space,
            'user_request': self.user_request_space,
        })

        # 定义动作空间，两个动作
        # 1. 对于每个服务器，部署哪些容器,x_{n,s}={0,1}表示服务器n上是否部署容器s？
        # 似乎就是上面的东西？
        self.now_container_place_space = spaces.MultiBinary((config['server_number'], config['container_number']))
        # 2. 对于每个请求，路由到哪个服务器？
        self.now_request_routing_space = spaces.MultiBinary((config['request_number'], config['server_number']))

        # 定义初始状态
        # 初始状态所有都是空的，
        # 时间戳决定了到达的请求的集合
        self.state = {
            'timestamp': 0,
        }

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # 传入一个动作，根据系统当前的状态，输出下一个观察空间
        # 至于这个动作如何选择在其他模块实现

        # 要返回5个变量，分别是 state (观察状态), reward, terminated, truncated, info
        reward = 0
        terminated = False
        truncated = False
        info = {}
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state += 1

        # 奖励为当前的位置，目标是尽可能向右移动
        reward = self.state

        # 如果达到最右边，则结束
        done = (self.state == 9)

        # 状态应该返回给智能体
        return self.state, reward, done, {}

    def reset(self):
        # 重置位置到最左边
        self.state = 0
        return self.state
