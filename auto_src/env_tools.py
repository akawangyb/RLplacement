import os.path

import numpy as np
import pandas as pd


def get_user_request_info(timestamp, user_number,father_dir):
    """
env_config.yaml
    :param user_number:
    :return: 返回一个三维数组，第一维是时间戳，第二维是容器id，第三维表示资源和延迟
    """
    user_request_info = np.zeros((timestamp, user_number, 5))  # 5分别是cpu,mem,net-in,net-out,lat
    path = os.path.join(father_dir, 'user_request_info.csv')
    # key = 'timestamp'
    data_frame = pd.read_csv(path)

    for index, row in data_frame.iterrows():
        # 第一个key是ts
        # ts_request = []
        if index >= timestamp:
            break
        for user_id in range(user_number):
            user_request_info[index, user_id, 0] = data_frame.loc[index, 'user_' + str(user_id) + '_cpu']
            user_request_info[index, user_id, 1] = data_frame.loc[index, 'user_' + str(user_id) + '_mem']
            user_request_info[index, user_id, 2] = data_frame.loc[index, 'user_' + str(user_id) + '_net-in']
            user_request_info[index, user_id, 3] = data_frame.loc[index, 'user_' + str(user_id) + '_net-out']
            user_request_info[index, user_id, 4] = data_frame.loc[index, 'user_' + str(user_id) + '_lat']

    return user_request_info


def get_server_info(server_number, father_dir):
    """
    :return:返回一个二维数组，第一维是 服务器id，第二维表示资源
    """
    server_info = np.zeros((server_number, 5))  # 5分别是cpu,mem,net-in,net-out,storage
    path = os.path.join(father_dir, 'server_info.csv')
    data_frame = pd.read_csv(path)
    for server_id, row in data_frame.iterrows():
        server_info[server_id][0] = data_frame.loc[server_id, 'cpu_size']
        server_info[server_id, 1] = data_frame.loc[server_id, 'mem_size']
        server_info[server_id, 2] = data_frame.loc[server_id, 'net-in_size']
        server_info[server_id, 3] = data_frame.loc[server_id, 'net-out_size']
        server_info[server_id, 4] = data_frame.loc[server_id, 'storage_size']
    return server_info


def get_container_info(container_number, father_dir):
    """
    :return:返回一个二维数组，第一维是 容器id，第二维表示资源
    """
    container_info = np.zeros((container_number, 2))
    path = os.path.join(father_dir, 'container_info.csv')

    data_frame = pd.read_csv(path)
    for container_id, row in data_frame.iterrows():
        container_info[container_id, 0] = data_frame.loc[container_id, 'container_size']
        container_info[container_id, 1] = data_frame.loc[container_id, 'container_pulling_delay']
    return container_info
