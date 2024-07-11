# --------------------------------------------------
# 文件名: plot
# 创建时间: 2024/6/19 12:14
# 描述: 从日志中读取数据，用图片显示出来
# 作者: WangYuanbo
# --------------------------------------------------


import re

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def simple_moving_average(data, window_size):
    """
    计算简单滑动平均 (SMA)

    参数:
    - data: 输入数据列表
    - window_size: 滑动窗口大小

    返回:
    - sma: 简单滑动平均值列表
    """
    sma = []
    for i in range(len(data)):
        if i + 1 < window_size:
            sma.append(data[i])  # 前期数据不足窗口大小时填充None
        else:
            window = data[i - window_size + 1:i + 1]
            window_avg = sum(window) / window_size
            sma.append(window_avg)
    return sma


# 查看系统中可用的字体
# for font in fm.findSystemFonts():
#     print(fm.FontProperties(fname=font).get_name())

# 设置全局字体
# plt.rcParams['font.sans-serif'] = ['KaiTi']  # 使用楷体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'DejaVu Serif'
plt.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
plt.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'


# 定义一个函数读取文件并返回数据
def read_data(file_path):
    with open(file_path, 'r') as file:
        log_data = file.read()
        rewards_data = re.findall(r'Episode: (\d+), total reward: (\d+\.\d+)', log_data)
    episodes = [int(data[0]) for data in rewards_data]
    total_rewards = [-float(data[1]) / 10000 for data in rewards_data]
    return episodes, total_rewards


# epoch
# file_path1 = r'log_res/epoch/imitation_learning_td3_env-20240707-083818/output_info.log'
# file_path2 = r'log_res/epoch/imitation_learning_td3_env-20240707-083850/output_info.log'
# file_path3 = r'log_res/epoch/imitation_learning_td3_env-20240707-083924/output_info.log'


# update interval
file_path1 = r'log_res/update_interval/imitation_learning_td3_env-20240707-211848/output_info.log'
file_path2 = r'log_res/update_interval/imitation_learning_td3_env-20240707-211912/output_info.log'
file_path3 = r'log_res/baseline/imitation_learning_td3_env-20240706-201411/output_info.log'

# 文件路径列表
file_paths = [file_path1, file_path2, file_path3]

label = ['update interval=10', 'update interval=30', 'update interval=20']
# 青色，橙色，砖红色，蓝色，紫色，灰色，黑色
color_set = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2', '#999999']
color = ['#FA7F6F', '#BEB8DC', '#82B0D2', ]
# 设置9:16比例的图形大小
fig, ax = plt.subplots(figsize=(4.8, 3.2))

# 对文件路径列表中的每个文件进行处理
for i, file_path in enumerate(file_paths):
    episodes, total_rewards = read_data(file_path)
    # 绘制total reward的折线图
    # ax.plot(episodes, simple_moving_average(total_rewards, 20), label=label[i], linewidth=1.2, color=color[i])
    ax.plot(episodes, total_rewards, label=label[i], linewidth=1.2, color=color[i])

# 添加图表标题和坐标轴标签
# 设置坐标轴刻度标签字体
# 指定罗马字体
# 设置x轴从0开始
ax.set_xlim(left=0)
roman_font_prop = FontProperties(family='Times New Roman')
kai_font_prop = FontProperties(family='KaiTi')
plt.xticks(fontproperties=roman_font_prop)  # X轴刻度标签使用楷体
plt.yticks(fontproperties=roman_font_prop)  # Y轴刻度标签使用楷体
# 添加数量级表示
ax.annotate(r'$\times 10^4$', xy=(0, 1.01), xycoords='axes fraction', size=9, fontproperties=roman_font_prop)
# ax.yaxis.get_offset_text().set(size=28, family='Times New Roman') #y 调节左上角数量级字体大小

ax.set_xlabel('训练轮次（回合）', size=10, fontproperties=kai_font_prop)
ax.set_ylabel('累计奖励（毫秒）', size=10, fontproperties=kai_font_prop)

# 添加图例
plt.legend()

# 设置图例字体
# ax.legend(prop=roman_font_prop, loc='upper right', bbox_to_anchor=(0.5, 0.5))
ax.legend(prop=roman_font_prop)

# 调整布局以减少上方的空白
plt.subplots_adjust(top=0.93, bottom=0.15, left=0.10, right=0.95)
# 或者使用紧密布局
# fig.tight_layout()

# 灰色的虚线网格，线条粗细为 0.5
ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

# 显示图形
plt.show()
