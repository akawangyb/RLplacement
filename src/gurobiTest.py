# --------------------------------------------------
# 文件名: gurobiTest
# 创建时间: 2024/2/18 23:29
# 描述: 假设在t时刻使用gurobi获得一个部署优化
# 作者: WangYuanbo
# --------------------------------------------------
import os

# print(config['server_number'])
# print(config['request_number'])
# print(config['server_info'])
# print(config['container_info'])
# print(config['user_request_info'][1])
# 先要定义变量
import gurobipy as gp
import numpy as np
import pandas as pd
import yaml
from gurobipy import GRB, quicksum

from placingENV import CustomEnv

# 'server_info.csv': 保存边缘服务器的信息
# 'container_info.csv': 保存待部署的容器信息
# 'user_request_info.csv': 保存每时刻的用户请求信息
# import pandas as pd
# 从csv文件读取数据
# 修改数据帧的索引
# # CSV文件名称
# csv_file = '../data/server_info.csv'  # 请用你实际的文件名替换 'your_file.csv'

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
# 创建模型
m = gp.Model("example")

# 定义变量的尺寸，例如我们这里使用3x3的二维数组
x_rows = range(env.server_number)
x_columns = range(env.container_number)

y_rows = range(env.request_number)

y_columns = range(env.server_number + 1)

# 添加二维0-1变量。lb=0, ub=1和vtype=GRB.BINARY指定变量为0-1变量
x = m.addVars(x_rows, x_columns, lb=0, ub=1, vtype=GRB.BINARY, name="x")
y = m.addVars(y_rows, y_columns, lb=0, ub=1, vtype=GRB.BINARY, name="y")

# # 打印变量以验证
# for i in rows:
#     for j in columns:
#         print(x[i, j])
# 添加约束

# print(x.shape)
# print(y.shape)
ts = 1
# 约束一
# 检查是不是所有的用户请求都有一个服务器完成
# 就是每一行的和为1
# 添加约束: 每一行的和应该等于1
for u in y_rows:
    m.addConstr(sum(y[u, n] for n in y_columns) == 1)

# # print(y)
# 约束二
# 检查请求路由的边缘服务器是否部署了对应的服务
for u in range(env.request_number):
    for n in range(env.server_number):
        col_name = 'user_' + str(u) + '_image'
        s = env.user_request_info[ts][col_name]
        m.addConstr(y[u, n] <= x[n, s])

# 约束三
# 对于服务部署，检查容器的的磁盘空间是不是满足的
for n in range(env.server_number):
    # 计算服务器n上的存储空间
    n_storage = 0
    for s in range(env.container_number):
        n_storage += x[n, s] * env.container_info[s]['container_size']
    m.addConstr(n_storage <= env.server_info[n]['storage_size'])

# 约束四
# 对于请求路由，首先检查服务器的cpu资源是不是足够的
for n in range(env.server_number):
    n_cpu = 0
    for u in range(env.request_number):
        col_name = 'user_' + str(u) + '_cpu'
        n_cpu += y[u, n] * env.user_request_info[ts][col_name]
    m.addConstr(n_cpu <= env.server_info[n]['cpu_size'])

# 更新模型以添加约束
m.update()

# 打印模型以验证
m.printStats()

# 假设最大化边缘服务的部署数量
objective = quicksum(y[u, n] for u in range(env.request_number) for n in range(env.server_number))
m.setObjective(objective, GRB.MAXIMIZE)

# Optimize model
m.optimize()

# 输出最优解的目标函数值
print('最优解的目标函数值: ', m.objVal)

x_result = np.array([[x[i, j].x for j in x_columns] for i in x_rows])
y_result = np.array([[y[i, j].x for j in y_columns] for i in y_rows])
print(x_result)
# for v in m.getVars():
#     print('%s: %g' % (v.varName, v.x))
