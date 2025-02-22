# --------------------------------------------------
# 文件名: plot
# 创建时间: 2024/6/19 12:14
# 描述: 从日志中读取数据，用图片显示出来
# 作者: WangYuanbo
# --------------------------------------------------
import re

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from tools import find_log_folders, find_name_log_folders


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


# 定义一个函数读取文件并返回数据
def read_data(file_path):
    with open(file_path, 'r') as file:
        log_data = file.read()
        rewards_data = re.findall(r'Episode: (\d+), total reward: (\d+\.\d+)', log_data)
    episodes = [int(data[0]) for data in rewards_data]
    total_rewards = [-float(data[1]) / 100000 for data in rewards_data]
    return episodes, total_rewards


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
color = ['#8ECFC9', '#FFBE7A', '#FA7F6F',
         '#82B0D2', '#BEB8DC', '#E7DAD2',
         '#F3D266', '#E7EFFA', '#999999',
         '#F5EBAE', '#EF8B67', '#992224',
         '#8074C8', '#D6EFF4', '#D8B365',
         '#5BB5AC', '#DE526C', '#6F6F6F',
         '#DD7C4F', '#6C61AF', '#B54764',
         '#f2fafc']
# color = ['#FA7F6F', '#BEB8DC', '#82B0D2', '#FFBE7A', '#8ECFC9', '#999999']

# baseline
# father_dir = r'log_res/temp/4exp_3'
# file_paths = find_name_log_folders(father_dir, agent_name='imitation_learning')
# label = [ele[-6:] for ele in file_paths]
father_dir = r'log_res/4exp_3'
file_paths = find_log_folders(father_dir)
label = ['TD3', 'DDPG', 'BC-DDPG', ]

print(file_paths)

# 文件路径列表

file_paths = [ele + '/output_info.log' for ele in file_paths]

# 设置9:16比例的图形大小
fig, ax = plt.subplots(figsize=(16, 9))
plt.xticks(fontproperties=roman_font_prop)
plt.yticks(fontproperties=roman_font_prop)

epoch = 1000
# 对文件路径列表中的每个文件进行处理
for i, file_path in enumerate(file_paths):
    # if i>=2:
    #     break
    episodes, total_rewards = read_data(file_path)
    total_rewards = sliding_average(total_rewards, 5)
    # 绘制total reward的折线图
    ax.plot(episodes[:epoch], total_rewards[:epoch], label=label[i], linewidth=3.0, color=color[i])
    # ax.plot(episodes[:epoch], total_rewards[:epoch],  linewidth=3.0, color=color[i])

# 添加图表标题和坐标轴标签
# 设置坐标轴刻度标签字体
# 指定罗马字体
# 设置x轴从0开始
ax.set_xlim(0, epoch)
plt.xticks(fontproperties=roman_font_prop)
plt.yticks(fontproperties=roman_font_prop)
# 添加数量级表示
ax.annotate(r'$\times 10^5$', xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)

ax.set_xlabel('训练轮次（回合）', fontproperties=kai_font_prop)
ax.set_ylabel('累计奖励（毫秒）', fontproperties=kai_font_prop)

ax.legend(prop=roman_font_prop, loc='lower right', ncol=1)

# 调整布局以减少上方的空白
plt.subplots_adjust(top=0.95, bottom=0.10, left=0.08, right=0.95)

# 灰色的虚线网格，线条粗细为 0.5
ax.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.8)

# # 创建放大的局部子图
# axins = inset_axes(ax, width="40%", height="50%", loc='lower left',
#                    bbox_to_anchor=(0.15, 0.25, 2.0, 1),
#                    bbox_transform=ax.transAxes,
#                    )
# for i, file_path in enumerate(file_paths):
#     episodes, total_rewards = read_data(file_path)
#     total_rewards = sliding_average(total_rewards, 10)
#     # 绘制total reward的折线图
#     axins.plot(episodes, total_rewards, linewidth=3.0, color=color[i])

# #  框的范围
# xlim0 = 0
# xlim1 = epoch
# ylim0 = -102
# ylim1 = -96
#
# # ylim0 = -0.5
# # ylim1 = -1.5
# # 调整子坐标系的显示范围
# sub_ax_xlim0 = xlim0
# sub_ax_xlim1 = xlim1
# sub_ax_ylim0 = ylim0
# sub_ax_ylim1 = ylim1
# axins.set_xlim(sub_ax_xlim0, sub_ax_xlim1)
# axins.set_ylim(sub_ax_ylim0, sub_ax_ylim1)
#
# # 原图中画方框
# tx0 = xlim0
# tx1 = xlim1
# ty0 = ylim0
# ty1 = ylim1
# sx = [tx0, tx1, tx1, tx0, tx0]
# sy = [ty0, ty0, ty1, ty1, ty0]
# ax.plot(sx, sy, linewidth=2.0, color='black', linestyle='--')
#
# # 画两条线
# xy = (xlim0, ylim1)
# xy2 = (xlim0, sub_ax_ylim1)
# con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
#                       axesA=axins, axesB=ax)
# con.set_linestyle('--')
# con.set_linewidth(1.5)
# axins.add_artist(con)
#
# xy = (xlim1, ylim1)
# xy2 = (xlim1, sub_ax_ylim1)
# con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
#                       axesA=axins, axesB=ax)
# # 设置连接线的粗细为2.0
# con.set_linewidth(1.5)
# # 设置连接线为虚线
# con.set_linestyle('--')
# axins.add_artist(con)
# axins.set_xlim(xlim0, xlim1)
# axins.annotate(r'$\times 10^5$', xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop, size=21)
# axins.grid(True, linestyle='--', linewidth=1.0, color='gray')
# plt.xticks(fontproperties=roman_font_prop,size=21)
# plt.yticks(fontproperties=roman_font_prop,size=21)
# #
# # 显示图形
plt.show()
