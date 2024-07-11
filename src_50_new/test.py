# --------------------------------------------------
# 文件名: plot
# 创建时间: 2024/6/19 12:14
# 描述: 从日志中读取数据，用图片显示出来
# 作者: WangYuanbo
# --------------------------------------------------
import re

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置全局字体
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
    total_rewards = [-float(data[1]) / 100000 for data in rewards_data]
    return episodes, total_rewards


# epoch
# file_path1 = r'log_res/epoch/imitation_learning_td3_env-20240711-000801/output_info.log'
# file_path2 = r'log_res/baseline/imitation_learning_td3_env-20240710-232954/output_info.log'
# file_path3 = r'log_res/epoch/imitation_learning_td3_env-20240711-000818/output_info.log'
# file_path4 = r'log_res/epoch/imitation_learning_td3_env-20240711-000837/output_info.log'
# label = ['Epoch=5', 'Epoch=10', 'Epoch=20', 'Epoch=40']

# update interval
# file_path1 = r'log_res/baseline/imitation_learning_td3_env-20240710-232954/output_info.log'
# file_path2 = r'log_res/update_interval/imitation_learning_td3_env-20240710-224124/output_info.log'
# file_path3 = r'log_res/update_interval/imitation_learning_td3_env-20240710-224142/output_info.log'
# file_path4 = r'log_res/update_interval/imitation_learning_td3_env-20240710-224159/output_info.log'
# label = ['Update Frequency=10', 'Update Frequency=20', 'Update Frequency=30', 'Update Frequency=40']

# learning rate
file_path1 = r'log_res/learning_rate/imitation_learning_td3_env-20240711-105622/output_info.log'
file_path2 = r'log_res/learning_rate/imitation_learning_td3_env-20240711-105628/output_info.log'
file_path3 = r'log_res/learning_rate/imitation_learning_td3_env-20240711-105635/output_info.log'
label = ['Learning Rate=0.001', 'Learning Rate=0.0001', 'Learning Rate=0.0005']

# batch_size
# file_path1 = r'log_res/batch_size/imitation_learning_td3_env-20240708-002111/output_info.log'
# file_path2 = r'log_res/batch_size/imitation_learning_td3_env-20240708-002037/output_info.log'
# file_path3 = r'log_res/baseline/imitation_learning_td3_env-20240706-201411/output_info.log'
# label = ['Batch Size=64', 'Batch Size=128', 'Batch Size=256']

# file_path1 = r'log_res/gamma/imitation_learning_td3_env-20240710-230440/output_info.log'
# file_path2 = r'log_res/gamma/imitation_learning_td3_env-20240710-232954/output_info.log'
# file_path3 = r'log_res/gamma/imitation_learning_td3_env-20240710-230458/output_info.log'
# file_path4 = r'log_res/gamma/imitation_learning_td3_env-20240710-230515/output_info.log'
# label = ['Gamma=0.90', 'Gamma=0.95', 'Gamma=0.97', 'Gamma=0.99']

# color = ['#FA7F6F', '#BEB8DC', '#82B0D2', '#FFBE7A']

# 文件路径列表
# file_paths = [file_path1, file_path2, file_path3, file_path4]
file_paths = [file_path1, file_path2, file_path3]
# file_paths = [file_path1, file_path2, file_path3]

# 青色，橙色，砖红色，蓝色，紫色，灰色，黑色
color_set = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2', '#999999']
color = ['#FA7F6F', '#BEB8DC', '#82B0D2', '#FFBE7A', '#8ECFC9', '#999999']
roman_font_prop = FontProperties(family='Times New Roman', size=9)
kai_font_prop = FontProperties(family='KaiTi')

# 设置9:16比例的图形大小
fig, ax = plt.subplots(figsize=(4.8, 3.2))
plt.xticks(fontproperties=roman_font_prop, size=9)
plt.yticks(fontproperties=roman_font_prop, size=9)
# 创建放大的局部子图
# axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
#                    bbox_to_anchor=(0.2, 0.4, 1.8, 1),
#                    bbox_transform=ax.transAxes)

# 对文件路径列表中的每个文件进行处理
for i, file_path in enumerate(file_paths):
    episodes, total_rewards = read_data(file_path)
    # 绘制total reward的折线图
    ax.plot(episodes[:500], total_rewards[:500], label=label[i], linewidth=1.2, color=color[i])
    # axins.plot(episodes, total_rewards, linewidth=1.2, color=color[i])

# 添加图表标题和坐标轴标签
# 设置坐标轴刻度标签字体
# 指定罗马字体
# 设置x轴从0开始
ax.set_xlim(left=0)

plt.xticks(fontproperties=roman_font_prop, size=9)
plt.yticks(fontproperties=roman_font_prop, size=9)
# 添加数量级表示
ax.annotate(r'$\times 10^5$', xy=(0, 1.01), xycoords='axes fraction', size=9, fontproperties=roman_font_prop)
# axins.annotate(r'$\times 10^4$', xy=(0, 1.01), xycoords='axes fraction', size=8, fontproperties=roman_font_prop)
# ax.yaxis.get_offset_text().set(size=28, family='Times New Roman') #y 调节左上角数量级字体大小

ax.set_xlabel('训练轮次（回合）', size=10, fontproperties=kai_font_prop)
ax.set_ylabel('累计奖励（毫秒）', size=10, fontproperties=kai_font_prop)

ax.legend(prop=roman_font_prop, fontsize=3, loc='lower right')

# 或者使用紧密布局
# fig.tight_layout()
# 调整布局以减少上方的空白
plt.subplots_adjust(top=0.93, bottom=0.15, left=0.13, right=0.95)

# 灰色的虚线网格，线条粗细为 0.5
ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

# xlim0 = 100
# # xlim0 = 500
# xlim1 = 500
# ylim0 = -5.8
# ylim1 = -4.5
# # ylim0 = -0.5
# # ylim1 = -1.5
# # 调整子坐标系的显示范围
# sub_ax_xlim0 = xlim0
# sub_ax_xlim1 = xlim1
# sub_ax_ylim0 = -5.8
# sub_ax_ylim1 = -4.5
# axins.set_xlim(sub_ax_xlim0, sub_ax_xlim1)
# axins.set_ylim(sub_ax_ylim0, sub_ax_ylim1)
#
# # # 原图中画方框
# tx0 = xlim0
# tx1 = xlim1
# ty0 = ylim0
# ty1 = ylim1
# sx = [tx0, tx1, tx1, tx0, tx0]
# sy = [ty0, ty0, ty1, ty1, ty0]
# ax.plot(sx, sy, linewidth=0.8, color='black', linestyle='--')
#
# # 画两条线
# xy = (xlim0, ylim1)
# xy2 = (xlim0, sub_ax_ylim1)
# con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
#                       axesA=axins, axesB=ax)
# con.set_linestyle('--')
# con.set_linewidth(0.8)
# axins.add_artist(con)
#
# xy = (xlim1, ylim1)
# xy2 = (xlim1, sub_ax_ylim1)
# con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
#                       axesA=axins, axesB=ax)
# # 设置连接线的粗细为2.0
# con.set_linewidth(0.8)
# # 设置连接线为虚线
# con.set_linestyle('--')
# axins.add_artist(con)
# axins.grid(True, linestyle='-.', linewidth=0.5, color='gray')

# 显示图形
plt.show()
