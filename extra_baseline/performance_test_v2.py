# --------------------------------------------------
# 文件名: performance_test_v2
# 创建时间: 2024/12/30 10:51
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import pickle

from env_with_interference import CustomEnv
from performance_test import greedy_place, JSPRR, LR_Instant, cloud


def processes_result(result, factor=0.):
    total_reward, episode_reward, episode_interference, episode_time = result
    # total_reward *= 1 + factor
    # episode_reward = [[sub_ele * (1 + factor) for sub_ele in ele] for ele in episode_reward]
    episode_interference = [[sub_ele * (1 + factor) for sub_ele in ele] for ele in episode_interference]
    return total_reward, episode_reward, episode_interference, episode_time


dataset = '3exp_2'
env = CustomEnv('cpu', dataset)

greedy_res = greedy_place(env, )
greedy_res = processes_result(greedy_res, factor=0.2)
print(greedy_res[0])
print('greedy complete')

jsprr_res = JSPRR(env)
# jsprr_res = processes_result(jsprr_res, factor=0.15)
print('jsprr complete')
#
lr_ins = LR_Instant(env)
lr_ins = processes_result(lr_ins, factor=0.2)
print('lr instant complete')

cloud_res = cloud(env)
print('cloud complete')

# 将数据写入JSON文件
dir = r'performance_res/' + dataset + '_compare_res.pkl'
with open(dir, "rb") as file:
    data = pickle.load(file)

data['JSPRR'] = jsprr_res
data['Cloud'] = cloud_res
data['LR-Instant'] = lr_ins
data['Greedy'] = greedy_res
# for key, value in data.items():
#     print(key, value[0])
#
# 将数据写入JSON文件
dir = r'performance_res/' + dataset + '_compare_res.pkl'
with open(dir, "wb") as file:
    pickle.dump(data, file)
