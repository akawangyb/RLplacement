# --------------------------------------------------
# 文件名: create_test
# 创建时间: 2024/7/15 0:24
# 描述: 生成50个用户请求，10个来自base，10个来自docker，10个来自air，10个来自city
# 作者: WangYuanbo
# --------------------------------------------------
import numpy as np
import pandas as pd

airport = [0.2261904761904762, 0.10119047619047619, 0.08928571428571429, 0.08928571428571429, 0.08928571428571429,
           0.20833333333333334, 0.23809523809523808, 0.26785714285714285, 0.23214285714285715, 0.19642857142857142,
           0.25595238095238093, 0.17261904761904762, 0.20238095238095238, 0.23214285714285715, 0.15476190476190477,
           0.14285714285714285, 0.21428571428571427, 0.19642857142857142, 0.7559523809523809, 1.0, 0.875,
           0.8869047619047619, 0.8571428571428571, 0.32142857142857145]
city = [0.19727891156462585, 0.10204081632653061, 0.08843537414965986, 0.10204081632653061, 0.10884353741496598,
        0.3197278911564626, 0.891156462585034, 0.8571428571428571, 0.7551020408163265, 0.8571428571428571, 1.0,
        0.42857142857142855, 0.3197278911564626, 0.2857142857142857, 0.22448979591836735, 0.23129251700680273,
        0.2108843537414966, 0.2585034013605442, 0.2789115646258503, 0.25170068027210885, 0.3129251700680272,
        0.3129251700680272, 0.2857142857142857, 0.36054421768707484]

base_mean = [0.26347084525744474, 0.21107754860258762, 0.1892642998467783, 0.1835062246698584, 0.18433621748815315,
             0.2228011971610012, 0.34659981346477986, 0.7428695096493871, 0.6115490834298121, 0.6429072495960113,
             0.6424403786357205, 0.6554090164215761, 0.6435556814853041, 0.6178777786693098, 0.6069581856536193,
             0.6614005270786415, 0.6830488888683164, 0.7337655259237165, 0.7048084104104415, 0.6560926488991449,
             0.6110451592187046, 0.5050765672079367, 0.4245672638333444, 0.551167105898868]

docker_mean = [0.025058150563893204, 0.022980010673378022, 0.15095126277191434, 0.07631893452993435,
               0.06296663475029575, 0.046381672918744635, 0.053690300166546814, 0.17675633897761592,
               0.10912342683064646, 0.13788006404026815, 0.13931367938502934, 0.0890287176284475, 0.4809638931149823,
               0.2106852354702976, 0.1995817440745981, 0.10316408461320775, 0.11493017133631786, 0.08910903221358818,
               0.04891158235067616, 0.0711185651420751, 0.12115455168472082, 0.05348951370369511, 0.020450101241446497,
               0.03794864147897288]


def generate_data(distribute, min_value, max_value):
    mean = 0
    if distribute == 'airport':
        mean = np.array(airport)
    elif distribute == 'city':
        mean = np.array(city)
    elif distribute == 'base':
        mean = base_mean
    elif distribute == 'docker':
        mean = docker_mean
    elif distribute == 'uniform':
        return np.random.uniform(low=min_value, high=max_value, size=24)
    res = np.random.normal(mean, 0.1, )
    res = np.clip(res, 0, 1)
    res = (max_value - min_value) * res + min_value
    assert res is not None
    return res


container_number = 40


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
        user_id = i // resource_category
        tag = 'base'
        if group[user_id] == 1:
            tag = 'airport'
        elif group[user_id] == 2:
            tag = 'city'
        elif group[user_id] == 3:
            tag = 'base'
        elif group[user_id] == 4:
            tag = 'docker'
        else:
            tag = 'uniform'

        if i % resource_category == 0:  # cpu 列
            pre_data[:, i] = generate_data(tag, min_cpu_frequency, max_cpu_frequency)
        elif i % resource_category == 1:  # mem 列
            pre_data[:, i] = generate_data(tag, min_mem_capacity, max_mem_capacity)
        elif i % resource_category == 2:  # net-in
            pre_data[:, i] = generate_data(tag, min_net_in_bandwith, max_net_in_bandwith)
        elif i % resource_category == 3:  # net-out
            pre_data[:, i] = generate_data(tag, min_net_out_bandwith, max_net_out_bandwith)
        elif i % resource_category == 4:  # lat
            pre_data[:, i] = generate_data(tag, min_lat, max_lat)

    # 生成具有特定列名的DataFrame
    user_request_df = pd.DataFrame(pre_data, columns=cols)

    # 插入timestamp列
    user_request_df.insert(0, 'timestamp', range(total_time))

    user_request_df.to_csv('user_request_info.csv', encoding='utf-8', index=False)


numbers = np.arange(container_number)
np.random.shuffle(numbers)
group = [0] * container_number
for i in range(container_number):
    if i < 10:
        group[numbers[i]] = 1
    elif i < 20:
        group[numbers[i]] = 1
    elif i < 30:
        group[numbers[i]] = 2
    elif i < 40:
        group[numbers[i]] = 2
    # else:
    #     group[numbers[i]] = 5
print(group)

create_user_request_data()
print('OK')
