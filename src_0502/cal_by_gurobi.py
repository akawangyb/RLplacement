# --------------------------------------------------
# 文件名: cal_by_gurobi
# 创建时间: 2024/4/28 16:38
# 描述: 用gurobi计算一个部署结果
# 作者: WangYuanbo
# --------------------------------------------------

# 先要定义变量
import gurobipy as gp
from gurobipy import GRB, quicksum

from env import CustomEnv
from tools import *


# 'server_info.csv': 保存边缘服务器的信息
# 'container_info.csv': 保存待部署的容器信息
# 'user_request_info.csv': 保存每时刻的用户请求信息


def model_assissted_solve(env: CustomEnv, timestamp: int):
    ts = timestamp
    # 创建模型
    m = gp.Model("example")
    m.setParam('OutputFlag', 0)
    # 定义变量的尺寸，例如我们这里使用3x3的二维数组
    x_rows = range(env.server_number)
    x_columns = range(env.container_number)

    y_rows = range(env.user_number)
    y_columns = range(env.server_number + 1)

    # 添加二维0-1变量。lb=0, ub=1和vtype=GRB.BINARY指定变量为0-1变量
    x = m.addVars(x_rows, x_columns, lb=0, ub=1, vtype=GRB.BINARY, name="x")
    y = m.addVars(y_rows, y_columns, lb=0, ub=1, vtype=GRB.BINARY, name="y")

    # 添加约束
    # 约束一
    # 检查是不是所有的用户请求都有一个服务器完成
    # 就是每一行的和为1
    # 添加约束: 每一行的和应该等于1
    for u in y_rows:
        m.addConstr(sum(y[u, n] for n in y_columns) == 1)

    # 约束二
    # 检查请求路由的边缘服务器是否部署了对应的服务
    for u in range(env.user_number):
        for n in range(env.server_number):
            imageID = env.user_request_info[ts][u][0].item()
            m.addConstr(y[u, n] <= x[n, imageID])

    # 约束三
    # 对于服务部署，检查容器的的磁盘空间是不是满足的
    for n in range(env.server_number):
        # 计算服务器n上的存储空间
        n_storage = 0
        for s in range(env.container_number):
            n_storage += x[n, s] * env.container_info[s][0]
        m.addConstr(n_storage <= env.server_info[n][4])

    # 约束四
    # 对于请求路由，首先检查服务器的cpu资源是不是足够的

    # 创建二维列表以保存线性表达式
    resource_demand = [[m.addVar() for _ in range(4)] for _ in range(env.server_number)]
    for n in range(env.server_number):
        for u in range(env.user_number):
            resource_demand[n][0] += y[u, n] * env.user_request_info[ts][u][1]
            resource_demand[n][1] += y[u, n] * env.user_request_info[ts][u][2]
            resource_demand[n][2] += y[u, n] * env.user_request_info[ts][u][3]
            resource_demand[n][3] += y[u, n] * env.user_request_info[ts][u][4]

    for n in range(env.server_number):
        for resource in range(4):
            m.addConstr(resource_demand[n][resource] <= env.server_info[n][resource])

    # 更新模型以添加约束
    m.update()

    # 打印模型以验证
    # m.printStats()

    # 假设最大化边缘服务的部署数量
    objective = quicksum(y[u, n] for u in range(env.user_number) for n in range(env.server_number))
    m.setObjective(objective, GRB.MAXIMIZE)

    # Optimize model
    m.optimize()

    # 输出最优解的目标函数值
    # print('最优解的目标函数值: ', m.objVal)

    x_result = np.array([[x[i, j].x for j in x_columns] for i in x_rows])
    y_result = np.array([[y[i, j].x for j in y_columns] for i in y_rows])

    return x_result, y_result


if __name__ == '__main__':
    env = CustomEnv('cpu')
    state, done = env.reset()
    return_list = []
    timestamp = 0
    while not done:
        x_result, y_result = model_assissted_solve(env, timestamp)
        placing_action = torch.tensor(x_result).int()
        routing_action = torch.tensor(y_result).int()
        routing_action = torch.argmax(routing_action, dim=1)
        env_action = {
            'placing_action': placing_action,
            'routing_action': routing_action
        }
        # print(env_action)
        next_state, reward, done, info = env.step(env_action)
        reward = torch.sum(reward).item()
        return_list.append(reward)
        timestamp += 1
    print(return_list)
    print(sum(return_list))
