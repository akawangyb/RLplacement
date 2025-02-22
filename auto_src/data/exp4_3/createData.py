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

container_number = 50
server_number = 4  # 假设m为10


# 随机出每个容器的信息
def create_container_data():
    # 假设下载带宽是1000 Mb/s
    downloading_bandwith = 2000

    container_df = pd.DataFrame({
        'container_id': range(container_number),
        'container_size': np.random.randint(200, 1000, container_number)   # 单位是MB
    })
    container_df['container_pulling_delay'] = container_df['container_size'] / downloading_bandwith * 1000
    container_df.to_csv('container_info.csv', encoding='utf-8', index=False)


# 随机出每个边缘服务器的信息
def create_edge_server_data():
    # 假设所有的服务器都是同构的
    min_cpu_frequency = 10
    max_cpu_frequency = 10
    min_mem_capacity = 16000  # 单位是MB
    max_mem_capacity = 16000
    min_net_bandwith = 500  # 单位是Mbits/s
    max_net_bandwith = 500

    min_disk_storage = 500000  # 单位是MB,存储空间设置为1T
    max_disk_storage = 500000
    server_df = pd.DataFrame({
        'server_id': range(server_number),
        'cpu_size': np.random.randint(min_cpu_frequency, max_cpu_frequency + 1, server_number),
        'mem_size': np.random.randint(min_mem_capacity, max_mem_capacity + 1, server_number),
        'net-in_size': np.random.randint(min_net_bandwith, max_net_bandwith + 1, server_number),
        'net-out_size': np.random.randint(min_net_bandwith, max_net_bandwith + 1, server_number),
        # 'read_size': np.random.randint(min_read_bandwith, max_read_bandwith + 1, server_number),
        # 'write_size': np.random.randint(min_write_bandwith, max_write_bandwith + 1, server_number),
        'storage_size': np.random.randint(min_disk_storage, max_disk_storage + 1, server_number)
    })
    server_df.to_csv('server_info.csv', encoding='utf-8', index=False)


# 随机出每个ts的用户请求
def create_user_request_data():
    """
    生成每个服务请求的信息
    请求哪个容器信息
    内存需求是100MB~4500MB
    存储需求是85MB~3350MB
    计算需求是0.5个cpu~28个CPU
    上行带宽的0~100 MBits/s
    下行带宽0~700MBits/s
    读带宽0~5MB/s
    写带宽0~10MB/s
    数据来源于2类，一类是真实的测试数据，一类是随机数据

    随机出每个ts的用户请求
    一共有n行，每行表示一个时间戳，
    每行的信息是user0_cpu,user0_mem,user0_net-in,user0_net-out,user0_read,user0_write,....
    """
    total_time = 24
    min_cpu_frequency = 0.1
    max_cpu_frequency = 0.5  # 最多20个
    min_mem_capacity = 50
    max_mem_capacity = 1000  # 最多10个+
    min_net_in_bandwith = 1
    max_net_in_bandwith = 20
    min_net_out_bandwith = 1
    max_net_out_bandwith = 20

    min_lat = 500
    max_lat = 5000

    # 也就是说生成一个二维表行是时间戳，列是用户，行是时间戳
    # 对于一个用户，他包含了两个信息，一是请求哪个服务，而是需要消耗多少资源
    # 初始化列名
    # cols = ['user_' + str(i // 2) + '_image' if i % 2 == 0 else 'user_' + str(i // 2) + '_cpu' for i in
    #         range(user_number * 2)]
    cols = []
    user_number = container_number
    for i in range(user_number):
        user_col = [f"user_{i}_cpu", f"user_{i}_mem", f"user_{i}_net-in", f"user_{i}_net-out",
                    f"user_{i}_lat"]
        cols.extend(user_col)

    resource_category = 5

    # 生成每一列的数据
    # 先预估请求数据的大小
    pre_data = np.empty((total_time, user_number * resource_category))

    for i in range(user_number * resource_category):
        # if i % resource_category == 0:  # 'request_image'列
        #     pre_data[:, i] = i / resource_category
        # elif i % resource_category == 1:  # cpu 列
        #     pre_data[:, i] = np.random.uniform(min_cpu_frequency, max_cpu_frequency, total_time)
        # elif i % resource_category == 2:  # mem 列
        #     pre_data[:, i] = np.random.uniform(min_mem_capacity, max_mem_capacity, total_time)
        # elif i % resource_category == 3:  # net-in
        #     pre_data[:, i] = np.random.uniform(min_net_in_bandwith, max_net_in_bandwith, total_time)
        # elif i % resource_category == 4:  # net-out
        #     pre_data[:, i] = np.random.uniform(min_net_out_bandwith, max_net_out_bandwith, total_time)
        # elif i % resource_category == 5:  # lat
        #     pre_data[:, i] = np.random.uniform(min_lat, max_lat, total_time)
        if i % resource_category == 0:  # cpu 列
            pre_data[:, i] = np.random.uniform(min_cpu_frequency, max_cpu_frequency, total_time)
        elif i % resource_category == 1:  # mem 列
            pre_data[:, i] = np.random.uniform(min_mem_capacity, max_mem_capacity, total_time)
        elif i % resource_category == 2:  # net-in
            pre_data[:, i] = np.random.uniform(min_net_in_bandwith, max_net_in_bandwith, total_time)
        elif i % resource_category == 3:  # net-out
            pre_data[:, i] = np.random.uniform(min_net_out_bandwith, max_net_out_bandwith, total_time)
        elif i % resource_category == 4:  # lat
            pre_data[:, i] = np.random.uniform(min_lat, max_lat, total_time)

    # 生成具有特定列名的DataFrame
    user_request_df = pd.DataFrame(pre_data, columns=cols)

    # 插入timestamp列
    user_request_df.insert(0, 'timestamp', range(total_time))

    user_request_df.to_csv('user_request_info.csv', encoding='utf-8', index=False)


def create_placing_info():
    # 创建一个m行n列的矩阵，每一行的前n/2个元素为1，后n/2个元素为0
    matrix = np.concatenate(
        (np.ones((server_number, container_number // 2)), np.zeros((server_number, container_number // 2))), axis=1)

    # 对每一行进行随机洗牌，打乱0和1的顺序
    for i in range(server_number):
        np.random.shuffle(matrix[i])

    columns = []
    for i in range(container_number):
        columns.append('container_' + str(i))
    # 创建一个DataFrame
    df = pd.DataFrame(matrix, columns=columns)

    # 插入timestamp列
    df.insert(0, 'server_id', range(server_number))
    # 将DataFrame保存为CSV文件
    df.to_csv('placing_info.csv', index=False)

    # print(matrix)


if __name__ == '__main__':
    create_container_data()
    create_edge_server_data()
    # create_user_request_data()
    print('hello')
