# --------------------------------------------------
# 文件名: plot
# 创建时间: 2024/6/19 12:14
# 描述: 从日志中读取数据，用图片显示出来
# 作者: WangYuanbo
# --------------------------------------------------
import pickle

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def sliding_average(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        if i < window_size:
            smoothed_data.append(data[i])
        else:
            window_values = data[i - window_size:i]
            average = sum(window_values) / window_size
            smoothed_data.append(average)
    return smoothed_data


# data 数据格式
# key:[total_reward, episode_reward, episode_interference, episode_time]
with open('model_para/performance_result.pkl', 'rb') as f:
    data = pickle.load(f)

# 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'DejaVu Serif'
plt.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
plt.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'
roman_font_prop = FontProperties(family='Times New Roman', size=25)
kai_font_prop = FontProperties(family='KaiTi', size=27)
son_font_prop = FontProperties(family='SimSun', size=23)

# 青色，橙色，砖红色，蓝色，紫色，灰色，黑色
color_set = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2', '#999999']
color = ['#FA7F6F', '#BEB8DC', '#82B0D2', '#FFBE7A', '#8ECFC9', '#999999']

# 设置9:16比例的图形大小
fig, ax = plt.subplots(figsize=(16, 9))

# 使用GridSpec对象设置子图的布局
# gs = GridSpec(1, 2, hspace=0.35)


# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
timestamp = 24
i = 0

for key, value in data.items():
    print(key, value[0])
    if key == 'Cloud':
        continue
    x_range = range(timestamp)
    episode_reward = value[1]
    episode_reward = [sum(ele) / 100000 for ele in episode_reward]
    episode_interference_factor = value[2]
    episode_interference_factor = [sum(ele) / 40 for ele in episode_interference_factor]
    #     # 绘制total reward的折线图
    #     # ax.plot(x_range, episode_reward, label=key, linewidth=3.0, color=color[i])
    ax.plot(x_range, episode_interference_factor, label=key, linewidth=3.0)
#     # ax2.plot(x_range, episode_reward, label=key, linewidth=3.0)
#     i += 1

# 添加图表标题和坐标轴标签
# 设置坐标轴刻度标签字体
# 指定罗马字体
# 设置x轴从0开始ax1 = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, timestamp - 1)
plt.xticks(range(timestamp), fontproperties=roman_font_prop)
plt.yticks(fontproperties=roman_font_prop)
# # 添加数量级表示
ax.annotate(r'$\times 10^5$', xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
ax.set_xlabel('时隙 ', fontproperties=kai_font_prop)
ax.set_ylabel('总时延（毫秒）', fontproperties=kai_font_prop)

ax.legend(prop=roman_font_prop, loc='upper left', ncol=3)
#
# # 调整布局以减少上方的空白
# plt.subplots_adjust(top=0.95, bottom=0.10, left=0.08, right=0.95)
#
# 灰色的虚线网格，线条粗细为 0.5
ax.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.8)

# 显示图形
plt.show()
