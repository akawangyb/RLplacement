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

# 也就是说对于容器镜像的表
# [image_id,image_size,downloading_time]
import numpy as np
import pandas as pd

container_number = 100


# 随机出每个容器的信息
def create_container_data():
    # 假设下载带宽是1000 Mb/s
    downloading_bandwith = 1000

    container_df = pd.DataFrame({
        'image_id': range(1, container_number + 1),
        'image_size': np.random.randint(70, 2000, container_number)
    })
    container_df['image_pulling_delay'] = container_df['image_size'] / downloading_bandwith
    container_df.to_csv('container_info.csv', encoding='utf-8', index=False)


# 随机出每个边缘服务器的信息
def create_edge_server_data():
    # 假设所有的服务器都是同构的
    min_cpu_frequency = 5
    max_cpu_frequency = 5
    min_disk_storage = 100
    max_disk_storage = 100
    server_number = 5
    server_df = pd.DataFrame({
        'server_id': range(1, server_number + 1),
        'cpu_size': np.random.randint(min_cpu_frequency, max_cpu_frequency, server_number),
        'storage_size': np.random.randint(min_disk_storage, max_disk_storage, server_number)
    })
    server_df.to_csv('server_info.csv', encoding='utf-8', index=False)


# 随机出每个ts的用户请求
def create_user_request_data():
    user_number = 500
    total_time = 100
    min_cpu_frequency = 0.1
    max_cpu_frequency = 0.5
    # 也就是说生成一个二维表行是时间戳，列是用户，行是时间戳
    # 对于一个用户，他包含了两个信息，一是请求哪个服务，而是需要消耗多少资源
    # 初始化列名
    cols = ['user_' + str(i // 2 + 1) + '_image' if i % 2 == 0 else 'user_' + str(i // 2 + 1) + '_cpu' for i in
            range(user_number * 2)]

    # 对于'request'列和'cpu'列生成不同的随机数
    # 假设'request'列中的随机数在0-1之间，'cpu'列中的随机数在100-200之间
    # 先预估一个数据的大小
    pre_data = np.empty((total_time, user_number * 2))

    for i in range(user_number * 2):
        if i % 2 == 0:  # 'request'列
            pre_data[:, i] = np.random.randint(1, container_number, total_time)
        else:  # 'cpu'列
            pre_data[:, i] = np.random.uniform(min_cpu_frequency, max_cpu_frequency, total_time)

    # 生成具有特定列名的DataFrame
    user_request_df = pd.DataFrame(pre_data, columns=cols)
    user_request_df.to_csv('user_request_info.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    create_container_data()
    # create_user_request_data()
    create_edge_server_data()
    create_user_request_data()
