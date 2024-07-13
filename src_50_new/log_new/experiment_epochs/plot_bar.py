# --------------------------------------------------
# 文件名: plot_bar
# 创建时间: 2024/7/12 11:52
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import re

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def read_data(file_path):
    with open(file_path, 'r') as file:
        log_data = file.read()
        rewards_data = re.findall(r'Episode: (\d+), total reward: (\d+\.\d+)', log_data)
    episodes = [int(data[0]) for data in rewards_data]
    total_rewards = [float(data[1]) / 100000 for data in rewards_data]
    return episodes, total_rewards


# 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'DejaVu Serif'
plt.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
plt.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'
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

# 数据
# episodes = list(range(21))
# total_rewards = [
#     493606.625, 490805.875, 493164.84375, 513401.25, 520861.625, 487377.625,
#     484696.375, 483008.5, 485044.8125, 498468.9375, 504570.09375, 473737.1875,
#     458921.125, 447227.25, 449096.71875, 452932.21875, 503021.75, 461740.6875,
#     450484.1875, 520009.6875, 507526.0
# ]

# 绘制折线图
fig, ax = plt.subplots(figsize=(16, 9))

file_path = r'experiment_epochs_env-20240712-122816/output_info.log'
episodes, total_rewards = read_data(file_path)

ax.plot(episodes, total_rewards, linewidth=3.0,marker='o', linestyle='-', color='skyblue',markersize=12)
# 添加数量级表示
ax.annotate(r'$\times 10^5$', xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
#
# 添加标题和标签
ax.set_xlim(0,30)
ax.set_xlabel('模仿学习轮次（回合）',  fontproperties=kai_font_prop)
ax.set_ylabel('总时延（毫秒）',  fontproperties=kai_font_prop)

plt.xticks(range(min(episodes), max(episodes) + 1, 3), fontproperties=roman_font_prop)
plt.yticks(fontproperties=roman_font_prop)
plt.subplots_adjust(top=0.95, bottom=0.10, left=0.08, right=0.95)

# 灰色的虚线网格，线条粗细为 0.5
ax.grid(True, linestyle='--', linewidth=1.5, color='gray',alpha=0.8)
# 显示图形
plt.show()
