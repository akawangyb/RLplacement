# --------------------------------------------------
# 文件名: data_tools
# 创建时间: 2024/2/26 15:30
# 描述: 公共工具类
# 作者: WangYuanbo
# --------------------------------------------------
import os
import pandas as pd


def get_user_request_info(user_number):
    user_request_info = []
    path = '../data/user_request_info.csv'
    key = 'timestamp'
    print(os.getcwd())
    data_frame = pd.read_csv(path)
    data_frame.set_index(key, inplace=True)
    data_dict = data_frame.to_dict('index')
    for ts, value in data_dict.items():
        # 第一个key是ts
        ts_request = []
        for index in range(user_number):
            ts_request.append({
                'cpu': value['user_' + str(index) + '_cpu'],
                'imageID': int(value['user_' + str(index) + '_image']),
            })
        user_request_info.append(ts_request)
    return user_request_info


def get_server_info():
    server_info = []
    path = '../data/server_info.csv'
    key = 'server_id'
    data_frame = pd.read_csv(path)
    data_frame.set_index(key, inplace=True)
    data_dict = data_frame.to_dict('index')
    for ts, value in data_dict.items():
        # 第一个key是ts
        ts_request = []
        # print(key, value)
        server_info.append({
            'cpu': value['cpu_size'],
            'storage': value['storage_size']
        })
    return server_info


def get_container_info():
    container_info = []
    path = '../data/container_info.csv'
    key = 'container_id'
    data_frame = pd.read_csv(path)
    data_frame.set_index(key, inplace=True)
    data_dict = data_frame.to_dict('index')
    for ts, value in data_dict.items():
        # 第一个key是ts
        ts_request = []
        container_info.append({
            'pulling': value['container_pulling_delay'],
            'storage': value['container_size']
        })
    return container_info


def get_placing_info():
    path = '../data/placing_info.csv'
    key = 'server_id'
    df = pd.read_csv(path, index_col=False)
    # 删除指定列
    df = df.drop(df.columns[0], axis=1)
    placing_info = df.values.tolist()

    return placing_info
