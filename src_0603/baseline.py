# --------------------------------------------------
# 文件名: baseline
# 创建时间: 2024/6/6 15:04
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import gurobipy as gp
import numpy as np
import torch
import torch.nn.functional as F
from gurobipy import GRB, quicksum

from env import CustomEnv


def baseline_cloud(env: CustomEnv):
    # 获得ts时刻的用户请求
    # 6分别是【ts】[user_id]imageID,cpu,mem,net-in,net-out,lat
    user_request_info = env.user_request_info  # 是一个张量
    lat = []
    for ts in range(env.end_timestamp):
        ts_lat = 0
        for user_id in range(env.user_number):
            ts_lat += user_request_info[ts][user_id][-1].item() + env.cloud_delay
        lat.append(ts_lat)

    return sum(lat)


def baseline_greedy_placement(env: CustomEnv):
    '''
    按照贪婪的模式去选择部署，
    :param env:
    :return:
    '''
    placed_position = [-1] * env.container_number  # 每个容器部署在什么地方
    server_storage_supply = env.server_info[:, 4]
    server_id = 0
    for container_id in range(env.container_number):
        storage_demand = env.container_info[container_id, 0]
        if server_storage_supply[server_id] >= storage_demand:
            server_storage_supply[server_id] -= storage_demand
            placed_position[container_id] = server_id
        else:
            server_id += 1
        if server_id >= env.server_number:
            break
    # 处理没放下的容器
    for container_id, server_id in enumerate(placed_position):
        if server_id == -1:
            placed_position[container_id] = env.server_number

    state, done = env.reset()
    action = torch.tensor(placed_position, dtype=torch.long)
    # action = torch.nn.functional.one_hot(action, env.server_number + 1)
    total_reward = torch.zeros(1)
    while not done:
        temp_action = action.clone().detach()
        onehot_action = F.one_hot(temp_action, env.server_number + 1)
        reward, is_valid_placing = env.cal_placing_rewards(placing_action=onehot_action)
        if not torch.all(is_valid_placing):
            temp_action[~is_valid_placing] = env.server_number
        onehot_action = F.one_hot(temp_action, env.server_number + 1)
        state, reward, done, info = env.step(onehot_action)
        total_reward += reward
    return total_reward.item()


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
        objective = quicksum(x[u, n] for u in range(env.container_number) for n in range(env.server_number))
        m.setObjective(objective, GRB.MAXIMIZE)

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
        total_reward += reward
    return total_reward.item()


if __name__ == '__main__':
    env = CustomEnv('cpu')
    res1 = baseline_greedy_placement(env=env)
    res2 = baseline_gurobi(env=env)
    res3 = baseline_cloud(env=env)
    print(res1)
    print(res2)
    print(res3)
