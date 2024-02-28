# --------------------------------------------------
# 文件名: createData
# 创建时间: 2024/2/11 23:29
# 描述: 给强化学习的环境生成随机数据
# 作者: WangYuanbo
# --------------------------------------------------
# 参数来源  TMC 2023
# Deep Reinforcement Learning Based Approach for Online Service Placement and Computation Resource Allocation in Edge Computing
# 假设有5个边缘服务器
# 每个边缘服务器的cpu资源是 10GHz
# 每个服务请求需要的cpu资源约为 [0.1,0.5]GHz
# 每个服务器的存储空间为100GB
# 容器镜像约为 [70,2000]MB 这个不对？

# 用户
# 一共有500个用户
# 每个用户的请求来自100个容器镜像库

import gurobipy as gp
# 也就是说对于容器镜像的表
# [image_id,image_size,downloading_time]
import numpy as np
import pandas as pd
import yaml
from gurobipy import GRB, quicksum

container_number = 10
server_number = 3  # 假设m为10
user_number = 10


# 随机出每个容器的信息
def create_container_data():
    # 假设下载带宽是1000 Mb/s
    downloading_bandwith = 100

    container_df = pd.DataFrame({
        'container_id': range(container_number),
        'container_size': np.random.randint(70, 2000, container_number)
    })
    container_df['container_pulling_delay'] = container_df['container_size'] / downloading_bandwith
    container_df.to_csv('container_info.csv', encoding='utf-8', index=False)


# 随机出每个边缘服务器的信息
def create_edge_server_data():
    # 假设所有的服务器都是同构的
    min_cpu_frequency = 5
    max_cpu_frequency = 6
    min_disk_storage = 100000
    max_disk_storage = 100001
    server_df = pd.DataFrame({
        'server_id': range(server_number),
        'cpu_size': np.random.randint(min_cpu_frequency, max_cpu_frequency, server_number),
        'storage_size': np.random.randint(min_disk_storage, max_disk_storage, server_number)
    })
    server_df.to_csv('server_info.csv', encoding='utf-8', index=False)


# 随机出每个ts的用户请求
def create_user_request_data():
    total_time = 100
    min_cpu_frequency = 0.1
    max_cpu_frequency = 0.5
    # 也就是说生成一个二维表行是时间戳，列是用户，行是时间戳
    # 对于一个用户，他包含了两个信息，一是请求哪个服务，而是需要消耗多少资源
    # 初始化列名
    cols = ['user_' + str(i // 2) + '_image' if i % 2 == 0 else 'user_' + str(i // 2) + '_cpu' for i in
            range(user_number * 2)]

    # 对于'request'列和'cpu'列生成不同的随机数
    # 假设'request'列中的随机数在0-1之间，'cpu'列中的随机数在100-200之间
    # 先预估一个数据的大小
    pre_data = np.empty((total_time, user_number * 2))

    for i in range(user_number * 2):
        if i % 2 == 0:  # 'request'列
            pre_data[:, i] = np.random.randint(0, container_number, total_time)
        else:  # 'cpu'列
            pre_data[:, i] = np.random.uniform(min_cpu_frequency, max_cpu_frequency, total_time)

    # 生成具有特定列名的DataFrame
    user_request_df = pd.DataFrame(pre_data, columns=cols)

    # 插入timestamp列
    user_request_df.insert(0, 'timestamp', range(total_time))

    user_request_df.to_csv('user_request_info.csv', encoding='utf-8', index=False)


def get_action_from_gurobi(config, timestamp):
    # 实际上我可以先计算好所有的时间t下的最优动作，
    m = gp.Model("example")
    m.setParam('OutputFlag', 0)

    # 定义变量的尺寸，例如我们这里使用3x3的二维数组
    x_rows = range(config['server_number'])
    x_columns = range(config['container_number'])

    y_rows = range(config['request_number'])

    y_columns = range(config['server_number'] + 1)

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
    for u in range(config['request_number']):
        for n in range(config['server_number']):
            col_name = 'user_' + str(u) + '_image'
            s = int(config['user_request_info'][ts][col_name])
            # print(type(s))
            m.addConstr(y[u, n] <= x[n, s])

    # 约束三，与时间戳无关
    # 对于服务部署，检查容器的的磁盘空间是不是满足的
    for n in range(config['server_number']):
        # 计算服务器n上的存储空间
        n_storage = 0
        for s in range(config['container_number']):
            n_storage += x[n, s] * config['container_info'][s]['container_size']
        m.addConstr(n_storage <= config['server_info'][n]['storage_size'])

    # 约束四，与时间戳有关
    # 对于请求路由，首先检查服务器的cpu资源是不是足够的
    for n in range(config['server_number']):
        n_cpu = 0
        for u in range(config['request_number']):
            col_name = 'user_' + str(u) + '_cpu'
            n_cpu += y[u, n] * config['user_request_info'][ts][col_name]
        m.addConstr(n_cpu <= config['server_info'][n]['cpu_size'])

    # 更新模型以添加约束
    m.update()

    # 打印模型以验证
    # m.printStats()

    # 假设最大化边缘服务的部署数量
    objective = quicksum(y[u, n] for u in range(config['request_number']) for n in range(config['server_number']))
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


def get_optimal_solutions():
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
        csv_path = file_name
        key = info_file[file_name]
        with open(csv_path, 'r', encoding='utf-8') as f:
            data_frame = pd.read_csv(f)
            data_frame.set_index(key, inplace=True)
            data_dict = data_frame.to_dict('index')
            config[file_name.replace('.csv', '')] = data_dict
    print(config['container_info'])
    optimal_solutions = []
    for ts in range(config['end_timestamp']):
        print(ts)
        optimal_solutions.append(get_action_from_gurobi(config=config, timestamp=ts))
        # print(optimal_solutions[-1])


def create_placing_info():


    # 创建一个m行n列的矩阵，每一行的前n/2个元素为1，后n/2个元素为0
    matrix = np.concatenate((np.ones((server_number, container_number // 2)), np.zeros((server_number, container_number // 2))), axis=1)

    # 对每一行进行随机洗牌，打乱0和1的顺序
    for i in range(server_number):
        np.random.shuffle(matrix[i])

    columns = []
    for i in range(container_number):
        columns.append('container_'+str(i))
    # 创建一个DataFrame
    df = pd.DataFrame(matrix,columns=columns)

    # 插入timestamp列
    df.insert(0, 'server_id', range(server_number))
    # 将DataFrame保存为CSV文件
    df.to_csv('placing_info.csv', index=False)

    # print(matrix)


if __name__ == '__main__':
    create_container_data()
    # # create_user_request_data()
    create_edge_server_data()
    create_user_request_data()
    # get_optimal_solutions()
    print('hello')
    create_placing_info()
