# --------------------------------------------------
# 文件名: gurobi_with_linear
# 创建时间: 2024/6/13 17:54
# 描述: 用线性回归预测干扰，然后用线性模型计算解
# 作者: WangYuanbo
# --------------------------------------------------
import pickle
import random

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB, quicksum
from sklearn.linear_model import LinearRegression

from env_with_interference import CustomEnv

# # 使用pickle库加载参数
with open('model_parameters.pkl', 'rb') as f:
    params = pickle.load(f)
#
weights = params["weights"]
bias = params["bias"]
model = LinearRegression()
model.coef_ = weights
model.intercept_ = bias


def baseline_gurobi(env: CustomEnv):
    """
    用gurobi去解这个问题，优化目标是最大化边缘请求数量
    :param env:
    """

    def model_solve():
        ts = env.timestamp
        # 创建模型
        m = gp.Model("example")
        m.setParam('OutputFlag', 0)
        # 定义变量的尺寸，例如我们这里使用3x3的二维数组
        x_rows = range(env.container_number)
        x_columns = range(env.server_number + 1)

        # 添加二维0-1变量。lb=0, ub=1和vtype=GRB.BINARY指定变量为0-1变量
        x = m.addVars(x_rows, x_columns, lb=0, ub=1, vtype=GRB.BINARY, name="x")

        # 添加约束
        # 约束一
        # 检查是不是所有的用户请求都有一个服务器完成
        for c in x_rows:
            m.addConstr(sum(x[c, n] for n in x_columns) == 1)

        # 约束三
        # 对于服务部署，检查容器的的磁盘空间是不是满足的
        for n in range(env.server_number):
            # 计算服务器n上的存储空间
            n_storage = 0
            for s in range(env.container_number):
                n_storage += x[s, n] * env.container_info[s][0]
            m.addConstr(n_storage <= env.server_info[n][4])

        # 约束四
        # 对于请求路由，首先检查服务器的cpu,mem,net资源是不是足够的
        resource_demand = [[m.addVar() for _ in range(4)] for _ in range(env.container_number)]
        for n in range(env.server_number):
            for u in range(env.container_number):
                resource_demand[n][0] += x[u, n] * env.user_request_info[ts][u][0]
                resource_demand[n][1] += x[u, n] * env.user_request_info[ts][u][1]
                resource_demand[n][2] += x[u, n] * env.user_request_info[ts][u][2]
                resource_demand[n][3] += x[u, n] * env.user_request_info[ts][u][3]

        for n in range(env.server_number):
            for resource in range(4):
                m.addConstr(resource_demand[n][resource] <= env.server_info[n][resource])

        # 更新模型以添加约束
        m.update()

        # 打印模型以验证
        # m.printStats()

        # 假设最大化边缘服务的部署数量
        # objective = quicksum(x[u, n] for u in range(env.container_number) for n in range(env.server_number))
        objective1 = quicksum(
            x[u, n] * env.edge_delay for u in range(env.container_number) for n in range(env.server_number))
        objective2 = quicksum(x[u, env.server_number] * env.cloud_delay for u in range(env.container_number))
        # delta = [[m.addVar() for _ in range(env.container_number)] for _ in range(env.server_number)]
        delta = [[m.addVar() for _ in range(env.server_number + 1)] for _ in range(env.container_number)]

        for u in range(env.container_number):
            for n in range(env.server_number):
                vector = [env.user_request_info[env.timestamp][u][0].item(),
                          env.user_request_info[env.timestamp][u][1].item(),
                          env.user_request_info[env.timestamp][u][2].item(),
                          env.user_request_info[env.timestamp][u][3].item(),
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
            x[u, n] * delta[u][n] for u in range(env.container_number) for n in range(env.server_number))
        objective = objective1 + objective2 + objective3
        m.setObjective(objective, GRB.MINIMIZE)

        # Optimize model
        m.optimize()

        # 输出最优解的目标函数值
        # print('最优解的目标函数值: ', m.objVal)

        x_result = np.array([[x[i, j].x for j in x_columns] for i in x_rows])

        return x_result

    env = CustomEnv('cpu')
    state, done = env.reset()
    total_reward = torch.zeros(1)
    while not done:
        action = model_solve()
        action = torch.tensor(action)
        state, reward, done, info = env.step(action)
        print(reward)

        reward = torch.sum(reward)
        total_reward += reward
    return total_reward


def baseline_gurobi_relax(env: CustomEnv):
    """
    用gurobi去解这个问题，优化目标是最大化边缘请求数量
    :param env:
    """

    def model_solve():
        ts = env.timestamp
        # 创建模型
        m = gp.Model("example")
        m.setParam('OutputFlag', 0)
        # 定义变量的尺寸，例如我们这里使用3x3的二维数组
        x_rows = range(env.container_number)
        x_columns = range(env.server_number + 1)

        # 添加二维0-1变量。lb=0, ub=1和vtype=GRB.BINARY指定变量为0-1变量
        x = m.addVars(x_rows, x_columns, lb=0, ub=1, name="x")

        # 添加约束
        # 约束一
        # 检查是不是所有的用户请求都有一个服务器完成
        for c in x_rows:
            m.addConstr(sum(x[c, n] for n in x_columns) == 1)

        # 约束三
        # 对于服务部署，检查容器的的磁盘空间是不是满足的
        for n in range(env.server_number):
            # 计算服务器n上的存储空间
            n_storage = 0
            for s in range(env.container_number):
                n_storage += x[s, n] * env.container_info[s][0]
            m.addConstr(n_storage <= env.server_info[n][4])

        # 约束四
        # 对于请求路由，首先检查服务器的cpu,mem,net资源是不是足够的
        resource_demand = [[m.addVar() for _ in range(4)] for _ in range(env.container_number)]
        for n in range(env.server_number):
            for u in range(env.container_number):
                resource_demand[n][0] += x[u, n] * env.user_request_info[ts][u][0]
                resource_demand[n][1] += x[u, n] * env.user_request_info[ts][u][1]
                resource_demand[n][2] += x[u, n] * env.user_request_info[ts][u][2]
                resource_demand[n][3] += x[u, n] * env.user_request_info[ts][u][3]

        for n in range(env.server_number):
            for resource in range(4):
                m.addConstr(resource_demand[n][resource] <= env.server_info[n][resource])

        # 更新模型以添加约束
        m.update()

        # 打印模型以验证
        # m.printStats()

        # 假设最大化边缘服务的部署数量
        # objective = quicksum(x[u, n] for u in range(env.container_number) for n in range(env.server_number))
        objective1 = quicksum(
            x[u, n] * env.edge_delay for u in range(env.container_number) for n in range(env.server_number))
        objective2 = quicksum(x[u, env.server_number] * env.cloud_delay for u in range(env.container_number))
        # delta = [[m.addVar() for _ in range(env.container_number)] for _ in range(env.server_number)]
        delta = [[m.addVar() for _ in range(env.server_number + 1)] for _ in range(env.container_number)]

        for u in range(env.container_number):
            for n in range(env.server_number):
                vector = [env.user_request_info[env.timestamp][u][0].item(),
                          env.user_request_info[env.timestamp][u][1].item(),
                          env.user_request_info[env.timestamp][u][2].item(),
                          env.user_request_info[env.timestamp][u][3].item(),
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
            x[u, n] * delta[u][n] for u in range(env.container_number) for n in range(env.server_number))
        objective = objective1 + objective2 + objective3
        m.setObjective(objective, GRB.MINIMIZE)

        # Optimize model
        m.optimize()

        # 输出最优解的目标函数值
        # print('最优解的目标函数值: ', m.objVal)

        x_result = np.array([[x[i, j].x for j in x_columns] for i in x_rows])

        return x_result

    def random_rand(action):
        # print(action)
        res = []
        for con in action:
            res_con = []
            for ele in con:
                # print(ele)
                p = 1 if random.random() >= ele else 0
                res_con.append(p)
            res.append(res_con)
        return res

    env = CustomEnv('cpu')
    state, done = env.reset()
    total_reward = torch.zeros(1)
    while not done:
        raw_action = model_solve()
        while True:
            action = random_rand(raw_action)
            action = torch.tensor(action)
            _, valid_placing = env.cal_placing_rewards(action)
            if valid_placing.all():
                break
        state, reward, done, info = env.step(action)
        # print(reward)

        reward = torch.sum(reward)
        total_reward += reward
    return total_reward


if __name__ == '__main__':
    # res = baseline_gurobi(CustomEnv('cpu'))
    res2 = baseline_gurobi_relax(CustomEnv('cpu'))
    print(res2)
