# --------------------------------------------------
# 文件名: placingENV
# 创建时间: 2024/2/12 0:55
# 描述: 服务部署的环境，只考虑存储和cpu
# 作者: WangYuanbo
# --------------------------------------------------
import os
from typing import Tuple

import gurobipy as gp
import gym
import numpy as np
import pandas as pd
import yaml
from gurobipy import quicksum, GRB
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

        # ########系统数据######
        # 定义一个非法动作的惩罚
        self.penalty = config['penalty']
        # 定义终止时间戳
        self.end_timestamp = config['end_timestamp']

        # 定义环境本身的配置，服务器数量，容器数量等。
        self.server_number = config['server_number']
        self.container_number = config['container_number']
        self.request_number = config['request_number']

        # ########随机生成的数据##########
        # 定义边缘服务器的参数存储和cpu空间
        self.server_info = config['server_info']

        # 定义每个用户的请求,需要哪个镜像?需要多少cpu计算资源，也是环境信息
        self.user_request_info = config['user_request_info']

        # 定义每个容器镜像信息,包括id,大小,拉取时延
        self.container_info = config['container_info']

        # ######定义观察空间#######
        # 1. 定义上一时刻的容器部署信息,x_{n,s}={0,1}表示服务器n上是否部署容器s
        # 此处用一个二维矩阵表示
        self.last_container_place_space = spaces.MultiBinary((self.server_number, self.container_number))

        # 2. 定义上一时刻的请求路由信息，y_{r_s,n}={0,1}关于服务s的请求是否由边缘服务器n完成。
        # 此处也用一个二维矩阵表示
        self.last_request_routing_space = spaces.MultiBinary((self.request_number, self.server_number + 1))

        # 3. 定义此时的用户服务请求
        # 假设请求数量是一定的，例如请求有500个，服务镜像100个。
        single_user_request_space = spaces.Discrete(self.container_number)
        self.user_request_space = spaces.Tuple([single_user_request_space] * self.request_number)

        # 定义最终的观察空间
        self.observation_space = spaces.Dict({
            'last_container_placement': self.last_container_place_space,
            'last_request_routing': self.last_request_routing_space,
            'user_request': self.user_request_space,
            # 'timestamp': self.timestamp_space,
        })
        # ##########动作空间##########
        # 定义动作空间，两个动作
        # 1. 对于每个服务器，部署哪些容器,x_{n,s}={0,1}表示服务器n上是否部署容器s？
        # 似乎就是上面的东西？
        self.now_container_place_space = spaces.MultiBinary((self.server_number, self.container_number))
        # 2. 对于每个请求，路由到哪个服务器？
        self.now_request_routing_space = spaces.MultiBinary((self.request_number, self.server_number + 1))
        self.action_space = spaces.Dict({
            'now_container_place': self.now_container_place_space,
            'now_request_routing': self.now_request_routing_space
        })
        # 定义动作空间的输入维度
        # 全部铺平后相加
        # self.action_dim = sum(np.prod(space.shape) for space in self.action_space.values())
        # 定义状态空间的输入维度
        self.state_dim = (np.prod(self.last_container_place_space.shape) +
                          np.prod(self.last_request_routing_space.shape) +
                          len(self.user_request_space))
        # 定义初始状态
        # 初始状态所有都是空的，
        # # 时间戳决定了到达的请求的集合
        self.state = {
            'last_container_placement': np.zeros((config['server_number'], config['container_number'])),
            'last_request_routing': np.zeros((config['request_number'], config['server_number'] + 1)),
            'user_request': np.zeros(config['request_number'])
        }
        self.timestamp = 1
        # 一旦环境初始化,直接求出每个时刻的最佳动作
        self.optimal_solution = self.get_optimal_solution()

    def get_action_from_gurobi(self, timestamp):
        # 实际上我可以先计算好所有的时间t下的最优动作，
        m = gp.Model("example")
        m.setParam('OutputFlag', 0)

        # 定义变量的尺寸，例如我们这里使用3x3的二维数组
        x_rows = range(self.server_number)
        x_columns = range(self.container_number)

        y_rows = range(self.request_number)

        y_columns = range(self.server_number + 1)

        # 添加二维0-1变量。lb=0, ub=1和vtype=GRB.BINARY指定变量为0-1变量
        x = m.addVars(x_rows, x_columns, lb=0, ub=1, vtype=GRB.BINARY, name="x")
        y = m.addVars(y_rows, y_columns, lb=0, ub=1, vtype=GRB.BINARY, name="y")

        ts = timestamp
        # 约束一,与时间戳无关
        # 检查是不是所有的用户请求都有一个服务器完成
        # 就是每一行的和为1
        # 添加约束: 每一行的和应该等于1
        for u in y_rows:
            m.addConstr(quicksum(y[u, n] for n in y_columns) == 1)
        # # print(y)
        # 约束二，与时间戳有关
        # 检查请求路由的边缘服务器是否部署了对应的服务
        for u in range(self.request_number):
            for n in range(self.server_number):
                col_name = 'user_' + str(u) + '_image'
                s = int(self.user_request_info[ts][col_name])
                # print(type(s))
                m.addConstr(y[u, n] <= x[n, s])

        # 约束三，与时间戳无关
        # 对于服务部署，检查容器的的磁盘空间是不是满足的
        for n in range(self.server_number):
            # 计算服务器n上的存储空间
            n_storage = 0
            for s in range(self.container_number):
                n_storage += x[n, s] * self.container_info[s]['container_size']
            m.addConstr(n_storage <= self.server_info[n]['storage_size'])

        # 约束四，与时间戳有关
        # 对于请求路由，首先检查服务器的cpu资源是不是足够的
        for n in range(self.server_number):
            n_cpu = 0
            for u in range(self.request_number):
                col_name = 'user_' + str(u) + '_cpu'
                n_cpu += y[u, n] * self.user_request_info[ts][col_name]
            m.addConstr(n_cpu <= self.server_info[n]['cpu_size'])

        # 更新模型以添加约束
        m.update()

        # 打印模型以验证
        # m.printStats()

        # 假设最大化边缘服务的部署数量
        objective = quicksum(y[u, n] for u in range(self.request_number) for n in range(self.server_number))
        m.setObjective(objective, GRB.MAXIMIZE)

        # Optimize model
        m.optimize()

        # 输出最优解的目标函数值
        print('最优解的目标函数值: {}'.format(m.objVal))

        # x_result = np.array([[round(x[i, j].x) for j in x_columns] for i in x_rows])
        # y_result = np.array([[round(y[i, j].x) for j in y_columns] for i in y_rows])
        x_result = np.array([[int(x[i, j].x) for j in x_columns] for i in x_rows])
        y_result = np.array([[int(y[i, j].x) for j in y_columns] for i in y_rows])
        placing_action = x_result
        routing_action = y_result
        action_dict = {'now_container_place': placing_action, 'now_request_routing': routing_action}
        return action_dict

    def get_optimal_solution(self):
        optimal_solution = []
        for ts in range(self.end_timestamp):
            solution = self.get_action_from_gurobi(timestamp=ts)
            optimal_solution.append(solution)
        return optimal_solution

    def get_request_from_ts(self, timestamp):
        ts = timestamp
        request_images = []
        for u in range(self.request_number):
            key = 'user_' + str(u) + '_image'
            request_images.append(self.user_request_info[ts][key])
        return np.array(request_images)

    # 执行一个动作，返回新的状态、奖励、以及是否完成
    # 至于这个动作是如何选择的，这里不需要管
    # 对于非法动作的处理：添加一个很大的负值作为惩罚
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # print(action)
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
        if not self.isValid(action, state):
            reward += self.penalty
            print("not valid action")
        else:
            reward += self.getReward(action, state)
            # 更新上一个部署系统状态
            state["last_container_placement"] = action["now_container_place"]
            state["last_request_routing"] = action["now_request_routing"]
        # 更新系统时间戳状态
        self.timestamp += 1
        ts = self.timestamp
        # 判断是否达到了截断条件和终止条件?
        if ts == self.end_timestamp:
            terminated = True
        else:  # 更新用户的请求信息
            state['user_request'] = self.get_request_from_ts(ts)
        done = terminated or truncated
        self.state = state
        return self.state, reward, done, info

    # 初始化环境状态
    def reset(self):
        # self.timestamp = 0
        # reset返回的状态要与obsSpace对应
        self.state = {'last_container_placement': np.zeros((self.server_number, self.container_number)),
                      'last_request_routing': np.zeros((self.request_number, self.server_number + 1)),
                      'user_request': self.get_request_from_ts(0)}
        self.timestamp = 0
        return self.state

    # 检查动作是不是合法的
    def isValid(self, action: ActType, state: ObsType) -> bool:
        x = action['now_container_place']
        y = action['now_request_routing']
        # print(x.shape)
        # print(y.shape)
        ts = self.timestamp
        flag = False
        # 约束一
        # 检查是不是所有的用户请求都有一个服务器完成
        for u in range(self.request_number):
            count_server = 0
            for n in range(self.server_number + 1):
                count_server += y[u][n]
            if count_server != 1:
                print('constraint 1')
                # print(y[u])
                return False
        # 约束二
        # 检查请求路由的边缘服务器是否部署了对应的服务
        for u in range(self.request_number):
            for n in range(self.server_number):
                s = int(state['user_request'][u])
                if y[u][n] > x[n][s]:
                    print('constraint 2')
                    return False

        # 约束三
        # 对于服务部署，检查容器的的磁盘空间是不是满足的
        for n in range(self.server_number):
            # 计算服务器n上的存储空间
            n_storage = 0
            for s in range(self.container_number):
                n_storage += x[n][s] * self.container_info[s]['container_size']
            if n_storage > self.server_info[n]['storage_size']:
                print('constraint 3')
                return False
        # 约束四
        # 对于请求路由，首先检查服务器的cpu资源是不是足够的
        for n in range(self.server_number):
            n_cpu = 0
            for u in range(self.request_number):
                col_name = 'user_' + str(u) + '_cpu'
                n_cpu += y[u][n] * self.user_request_info[ts][col_name]
            if n_cpu > self.server_info[n]['cpu_size']:
                print('constraint 4')
                return False

        return True
        # 需要考虑请求覆盖的问题
        # 这个约束在建模里面没有

    # 对于一个合法的动作计算他获得的奖励
    # 初始版本，只计算镜像拉取带来的延迟
    def getReward(self, action: ActType, state: ObsType) -> int:
        # 如果当前时间戳是0，直接计算拉取延迟
        x = action['now_container_place']
        b = self.container_info
        total_delay = 0
        delay_pulling = 0
        last_placing = state['last_container_placement']
        # if self.timestamp == 0:
        #     for n in range(self.server_number):
        #         for s in range(self.container_number):
        #             delay_pulling += x[n][s] * b[s]['container_pulling_info']
        #             print(delay_pulling)
        # 先对当前时间戳的时延求和，然后减去上个时间戳已有的
        for n in range(self.server_number):
            for s in range(self.container_number):
                delay_pulling += x[n][s] * b[s]['container_pulling_delay'] * (x[n][s] - last_placing[n][s])
                # print(delay_pulling)

        # for n in range(self.server_number):
        #     for s in range(self.container_number):
        #         delay_pulling -= x[n][s] * last_placing[n][s] * b[s]['container_pulling_delay']
        total_delay += delay_pulling
        return -total_delay


if __name__ == '__main__':
    # print('hello')
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    father_directory = '../data'
    info_file = {
        'server_info.csv': 'server_id',
        'container_info.csv': 'container_id',
        'user_request_info.csv': 'timestamp',
    }
    # import pandas as pd
    # 从csv文件读取数据
    # 修改数据帧的索引
    # # CSV文件名称
    # csv_file = '../data/server_info.csv'  # 请用你实际的文件名替换 'your_file.csv'
    for file_name in info_file:
        csv_path = os.path.join(father_directory, file_name)
        key = info_file[file_name]
        with open(csv_path, 'r', encoding='utf-8') as f:
            data_frame = pd.read_csv(f)
            data_frame.set_index(key, inplace=True)
            data_dict = data_frame.to_dict('index')
            # print(data_dict)
            config[file_name.replace('.csv', '')] = data_dict
    # print(config)
    env = CustomEnv(config=config)
    # print("hello")
    # print(env.action_space)
    # print(env.action_space.sample())
    # print(env.input_dim)
    # print(env.action_space.n)
    # print(env.observation_space)
    initial_state = env.reset()
    # print(env.optimal_solution)
    # print(env.timestamp)
    # print(initial_state['user_request'])
    # # print(initial_state)
    # print(env.action_dim)
    # print(env.state_dim)
