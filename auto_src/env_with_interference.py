# --------------------------------------------------
# 文件名: env
# 创建时间: 2024/6/3 21:28
# 描述: 新的环境，只有一个变量
# 作者: WangYuanbo
# --------------------------------------------------
import os
import pickle
from collections import namedtuple

import gurobipy as gp
import gym
import numpy as np
import torch
import yaml
from catboost import CatBoostRegressor
from gurobipy import quicksum, GRB
from gym import spaces
from torch import Tensor

with open('model_para/model_parameters.pkl', 'rb') as f:
    params = pickle.load(f)
#
weights = params["weights"]
bias = params["bias"]
from env_tools import get_server_info, get_container_info, get_user_request_info

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


class CustomEnv(gym.Env):
    # 定义动作空间和观察空间
    def __init__(self, device, data_dir):
        super().__init__()

        self.data_dir = data_dir
        father_dir = os.path.join('data', self.data_dir)
        data_path = os.path.join(father_dir, 'env_config.yaml')
        with open(data_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        config = Config(**config_data)

        self.device = device
        self.name = 'env'
        # ########系统数据######
        # 定义一个非法动作的惩罚
        self.penalty = config.penalty
        # 定义终止时间戳
        self.end_timestamp = config.end_timestamp

        # 其他信息
        self.max_cpu = config.max_cpu
        self.max_mem = config.max_mem
        self.max_storage = config.max_disk
        self.max_net_in = config.max_net_in
        self.max_net_out = config.max_net_out

        self.server_number = config.server_number
        self.container_number = config.container_number
        self.user_number = self.container_number

        self.cloud_delay = config.cloud_delay
        self.edge_delay = config.edge_delay

        # 定义请求信息和服务器信息
        self.user_request_info = torch.tensor(get_user_request_info(self.end_timestamp, self.user_number, father_dir),
                                              dtype=torch.float32).to(self.device)
        self.server_info = torch.tensor(get_server_info(self.server_number, father_dir),
                                        dtype=torch.float32).to(self.device)
        self.container_info = torch.tensor(get_container_info(self.container_number, father_dir),
                                           dtype=torch.float32).to(self.device)

        self.container_storage = self.container_info[:, 0].to(self.device).view(-1)  # 获得其第一列
        self.container_pulling_delay = self.container_info[:, 1].to(self.device).view(-1)  # 获得第二列

        self.placing_agents_number = self.server_number
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

        self.obs_space_dim = {
            'server': (self.server_number, 5),  # 服务器可以提供的资源，[cpu,mem,net-in,net-out,disk]
            'image_storage': (self.container_number, 2),  # 容器的属性 [store,lat]
            'user_request': (self.user_number, 5),  # t时刻的请求属性 [cpu,mem,net-in,net-out,lat]
            'last_placing_result': (self.container_number, self.server_number + 1),  # 上一时刻的部署结果，这是一个0，1矩阵
        }
        self.state_dim = 0
        for key, value in self.obs_space_dim.items():
            self.state_dim += value[0] * value[1]
        # 假设每个智能体都能看到环境的全部信息

        self.state_dims = self.state_dim

        # ##########动作空间##########
        # 定义placing动作空间
        placing_action_space = spaces.MultiBinary((self.container_number, self.server_number + 1))
        # 最终动作空间
        self.action_space = placing_action_space

        # 定义初始状态,reset里面去定义
        self.state = {}
        self.state_tensor = None
        self.timestamp = 0
        self.last_placing_state = None
        self.action_dim = (self.container_number, (self.server_number + 1))
        self.model = CatBoostRegressor()
        self.model.load_model(r'model_para/catboost_model.bin')

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
        last_placing_state = torch.zeros((self.container_number, self.server_number + 1),
                                         dtype=torch.float32).to(self.device)
        self.state = {  # 可读的系统状态
            "server_state": server_state,
            "container_state": container_state,
            "user_request_state": user_request_state,
            'last_placing_state': last_placing_state
        }

        self.state_tensor = torch.concat(  # 神经网络输入的系统状态
            (user_request_state.view(-1),
             container_state.view(-1),
             server_state.view(-1),
             last_placing_state.view(-1))
        ).to(self.device)
        return self.state_tensor.view(-1), torch.zeros(1).bool()

    def step(self, action: Tensor):
        # 传入动作，根据系统当前的状态，输出下一个观察空间
        # 至于这个动作是如何得到在其他模块实现
        # 要返回5个变量，分别是 state (观察状态), reward, terminated, truncated, info
        # 这里传入的动作，只有一个placing
        placing_action = action.clone().detach()
        assert placing_action.shape == (self.container_number, self.server_number + 1,), \
            f'Expected shape {(self.container_number, self.server_number + 1)}, but got {placing_action.shape}'

        placing_rewards, is_valid_placing_action = self.cal_placing_rewards(placing_action=placing_action)
        assert placing_rewards.shape[0] == self.container_number, \
            f'placing rewards shape {placing_rewards.shape}'

        # 更新系统状态
        now_placing_action = placing_action
        state = self.state
        # 更新请求状态
        self.timestamp += 1
        state['user_request_state'] = self.user_request_info[self.timestamp] \
            if self.timestamp < self.end_timestamp else torch.zeros_like(state['user_request_state'])
        now_placing_action[~is_valid_placing_action, :] = 0

        # 更新placing状态
        state['last_placing_state'] = now_placing_action
        self.state = state

        # 使用逻辑或“|”操作符来判断两个张量中是否有False
        done = torch.full((1,), False).to(self.device)
        if self.timestamp == self.end_timestamp:
            done = ~ done
        self.state_tensor = torch.concat(
            (self.state['user_request_state'].view(-1),
             self.state['container_state'].view(-1),
             self.state['server_state'].view(-1),
             self.state['last_placing_state'].view(-1)
             )

        ).to(self.device)
        # rewards = torch.sum(placing_rewards) if torch.all(is_valid_placing_action) else self.penalty
        # rewards = torch.sum(placing_rewards)
        rewards = placing_rewards
        # print('is_valid_placing_action', is_valid_placing_action)

        return self.state_tensor.view(-1), rewards, done, rewards

    def cal_placing_rewards(self, placing_action, interference_factor=False):
        """
        :param interference_factor: 是否输出干扰因子
        :param placing_action: 一个二维张量， container_number * server_number + 1
        :return: placing_rewards, is_valid_placing_action，【server_number】张量
        """
        # 1. 检查是否满足部署约束每个容器必须部署在一个服务器上
        row_sums = placing_action.sum(dim=1)

        # 检查每行的和是否恰好等于1
        constraint_1 = row_sums == torch.ones_like(row_sums)  # 如果是每一行的和都是1返回true，否则返回false

        # 2. 检查每个 server 的存储资源有没有超过

        server_action = placing_action.clone().float().to(self.device)[:, :-1].t()
        server_storage_demand = server_action * self.container_storage  #
        server_storage_demand = torch.sum(server_storage_demand, dim=1)
        is_valid_storage = server_storage_demand <= self.server_info[:, 4]  # 取出第五列比较

        # 3. 检查cpu资源有没有超过
        server_cpu_demand = server_action * self.user_request_info[self.timestamp, :, 0]
        server_cpu_demand = torch.sum(server_cpu_demand, dim=-1)
        is_valid_cpu = server_cpu_demand <= self.server_info[:, 0]

        # 检查mem
        server_mem_demand = server_action * self.user_request_info[self.timestamp, :, 1]
        server_mem_demand = torch.sum(server_mem_demand, dim=-1)
        is_valid_mem = server_mem_demand <= self.server_info[:, 1]

        # 检查net_in
        server_net_in_demand = server_action * self.user_request_info[self.timestamp, :, 2]
        server_net_in_demand = torch.sum(server_net_in_demand, dim=-1)
        is_valid_net_in = server_net_in_demand <= self.server_info[:, 2]

        # 检查net_out
        server_net_out_demand = server_action * self.user_request_info[self.timestamp, :, 3]
        server_net_out_demand = torch.sum(server_net_out_demand, dim=-1)
        is_valid_net_out = server_net_out_demand <= self.server_info[:, 3]

        is_valid_server = is_valid_net_in & is_valid_net_out & is_valid_cpu & is_valid_mem & is_valid_storage

        #  计算用户的路由是否合法
        user_action = torch.argmax(placing_action, dim=-1)

        extended_valid_server_action = torch.cat((is_valid_server, torch.Tensor([False])))
        valid_edge_mask = extended_valid_server_action[user_action].bool()

        cloud_routing_mask = (placing_action[:, -1] == 1) & constraint_1

        '''
        因此得到了3个mask，
        valid_edge_mask 合法的边缘路由
        cloud_routing_mask 路由到云的请求
        '''
        # 计算奖励
        # 获得路由到边的placing_action
        assert cloud_routing_mask.shape == valid_edge_mask.shape
        placing_rewards = torch.zeros(self.container_number).to(self.device)
        placing_rewards[valid_edge_mask] = self.edge_delay
        # 这里加上前后的镜像的拉取延迟

        last_placing = self.state['last_placing_state']
        last_user_action = torch.argmax(last_placing, dim=-1)
        now_user_action = user_action
        pulling_mask = (now_user_action != last_user_action) & valid_edge_mask
        placing_rewards[pulling_mask] += self.container_pulling_delay[pulling_mask]

        placing_rewards[cloud_routing_mask] = self.cloud_delay + \
                                              self.user_request_info[self.timestamp, cloud_routing_mask, -1]

        # 计算每个容器的干扰
        # 首先计算每个服务的部署资源总量
        server_resource_demand = torch.stack(
            [server_cpu_demand, server_mem_demand, server_net_in_demand, server_net_out_demand], dim=-1)
        # print(server_resource_demand.shape)
        server_resource_supply = self.server_info[:, :4]
        container_interference = torch.zeros_like(placing_rewards)
        for i in range(self.container_number):
            if valid_edge_mask[i]:
                server_id = user_action[i]
                container_resource_demand = self.user_request_info[self.timestamp, i, :4]

                demand = container_resource_demand.clone().view(-1).tolist()
                supply = (server_resource_supply[server_id] - server_resource_demand[server_id]).clone().view(
                    -1).tolist()
                input_vector = [demand[i] / supply[i] for i in range(4)] + [0, 0]
                assert len(input_vector) == 6, f'length of input_vector: {len(input_vector)}'
                container_interference[i] = self.model.predict(input_vector)

        placing_rewards[valid_edge_mask] += self.user_request_info[self.timestamp, valid_edge_mask, -1] * \
                                            (1 + container_interference[valid_edge_mask])

        valid_placing_action = valid_edge_mask | cloud_routing_mask

        placing_rewards[~valid_placing_action] = self.penalty

        if interference_factor:
            return placing_rewards, valid_placing_action, container_interference

        return placing_rewards, valid_placing_action

    def model_solve(self, relax=False):
        """
        :return: 不松驰直接求解
        """
        ts = self.timestamp
        # 创建模型
        m = gp.Model("example")
        # 设置解的池参数
        m.setParam('OutputFlag', 0)

        # 定义变量的尺寸，例如我们这里使用3x3的二维数组
        x_rows = range(self.container_number)
        x_columns = range(self.server_number + 1)

        # 添加二维0-1变量。lb=0, ub=1和vtype=GRB.BINARY指定变量为0-1变量
        type = GRB.CONTINUOUS if relax else GRB.BINARY
        x = m.addVars(x_rows, x_columns, lb=0, ub=1, vtype=type, name="x")

        # 添加约束
        # 约束一
        # 检查是不是所有的用户请求都有一个服务器完成
        for c in x_rows:
            m.addConstr(sum(x[c, n] for n in x_columns) == 1)
        # 约束三
        # 对于服务部署，检查容器的的磁盘空间是不是满足的
        for n in range(self.server_number):
            # 计算服务器n上的存储空间
            n_storage = 0
            for s in range(self.container_number):
                n_storage += x[s, n] * self.container_info[s][0]
            m.addConstr(n_storage <= self.server_info[n][4])

        # 约束四
        # 对于请求路由，首先检查服务器的cpu,mem,net资源是不是足够的
        resource_demand = [[m.addVar() for _ in range(4)] for _ in range(self.container_number)]
        for n in range(self.server_number):
            for u in range(self.container_number):
                resource_demand[n][0] += x[u, n] * self.user_request_info[ts][u][0]
                resource_demand[n][1] += x[u, n] * self.user_request_info[ts][u][1]
                resource_demand[n][2] += x[u, n] * self.user_request_info[ts][u][2]
                resource_demand[n][3] += x[u, n] * self.user_request_info[ts][u][3]

        for n in range(self.server_number):
            for resource in range(4):
                m.addConstr(resource_demand[n][resource] <= self.server_info[n][resource])

        # 更新模型以添加约束
        m.update()

        objective1 = quicksum(
            x[u, n] * self.edge_delay for u in range(self.container_number) for n in range(self.server_number))
        objective2 = quicksum(x[u, self.server_number] * self.cloud_delay for u in range(self.container_number))
        delta = [[m.addVar() for _ in range(self.server_number + 1)] for _ in range(self.container_number)]

        for u in range(self.container_number):
            for n in range(self.server_number):
                for u in range(self.container_number):
                    for n in range(self.server_number):
                        vector = [self.user_request_info[self.timestamp][u][0].item(),
                                  self.user_request_info[self.timestamp][u][1].item(),
                                  self.user_request_info[self.timestamp][u][2].item(),
                                  self.user_request_info[self.timestamp][u][3].item(),
                                  0,
                                  0,
                                  resource_demand[n][0],
                                  resource_demand[n][1],
                                  resource_demand[n][2],
                                  resource_demand[n][3],
                                  200,
                                  200
                                  ]
                        res = 0
                        for i in range(12):
                            res = weights[i] * vector[i]
                        res += bias
                        delta[u][n] += res

        objective3 = quicksum(
            x[u, n] * delta[u][n] for u in range(self.container_number) for n in range(self.server_number))
        objective = objective1 + objective2 + objective3
        m.setObjective(objective, GRB.MINIMIZE)

        # Optimize model
        m.optimize()

        return np.array([[x[i, j].x for j in x_columns] for i in x_rows])

    def model_solve_max_edge(self, relax=False):
        ts = self.timestamp
        # 创建模型
        m = gp.Model("example")
        m.setParam('OutputFlag', 0)
        x_rows = range(self.container_number)
        x_columns = range(self.server_number + 1)

        # 添加二维0-1变量。lb=0, ub=1和vtype=GRB.BINARY指定变量为0-1变量
        type = GRB.CONTINUOUS if relax else GRB.BINARY
        x = m.addVars(x_rows, x_columns, lb=0, ub=1, vtype=type, name="x")

        # 添加约束
        # 约束一
        # 检查是不是所有的用户请求都有一个服务器完成
        for c in x_rows:
            m.addConstr(sum(x[c, n] for n in x_columns) == 1)
        # 约束三
        # 对于服务部署，检查容器的的磁盘空间是不是满足的
        for n in range(self.server_number):
            # 计算服务器n上的存储空间
            n_storage = 0
            for s in range(self.container_number):
                n_storage += x[s, n] * self.container_info[s][0]
            m.addConstr(n_storage <= self.server_info[n][4])

        # 约束四
        # 对于请求路由，首先检查服务器的cpu,mem,net资源是不是足够的
        resource_demand = [[m.addVar() for _ in range(4)] for _ in range(self.container_number)]
        for n in range(self.server_number):
            for u in range(self.container_number):
                resource_demand[n][0] += x[u, n] * self.user_request_info[ts][u][0]
                resource_demand[n][1] += x[u, n] * self.user_request_info[ts][u][1]
                resource_demand[n][2] += x[u, n] * self.user_request_info[ts][u][2]
                resource_demand[n][3] += x[u, n] * self.user_request_info[ts][u][3]

        for n in range(self.server_number):
            for resource in range(4):
                m.addConstr(resource_demand[n][resource] <= self.server_info[n][resource])

        # 更新模型以添加约束
        m.update()

        # 打印模型以验证
        # m.printStats()

        # 假设最大化边缘服务的部署数量
        objective = quicksum(x[u, n] for u in range(self.container_number) for n in range(self.server_number))
        m.setObjective(objective, GRB.MAXIMIZE)

        # Optimize model
        m.optimize()
        x_result = np.array([[x[i, j].x for j in x_columns] for i in x_rows])

        return x_result


if __name__ == '__main__':
    env = CustomEnv('cpu')
    state, done = env.reset()
    # 生成一个 n*m 的随机整数张量，其中数值的范围在 [0, m)
    m = env.server_number + 1
    n = env.container_number
    random_tensor = torch.randint(m, (n,))
    # 对该张量进行 one-hot 编码
    action = torch.nn.functional.one_hot(random_tensor, num_classes=m)
    # action = [[0, 0, 1],
    #           [0, 1, 0],
    #           [0, 1, 0],
    #           [0, 0, 1],
    #           [0, 1, 0]]
    # action = torch.tensor(action)
    print(action)
    next_state, rewards, done, info = env.step(action)
    print('rewards', rewards)
