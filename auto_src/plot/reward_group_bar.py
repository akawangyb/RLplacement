# --------------------------------------------------
# 文件名: test
# 创建时间: 2024/7/20 15:57
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator

from tools import get_solutions_info

# # 设置全局字体
config = {
    "font.family": "serif",  # 使用衬线体
    "font.serif": ["SimSun"],  # 全局默认使用衬线宋体
    "font.size": 14,  # 五号，10.5磅
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
}
plt.rcParams.update(config)

roman_font_prop = FontProperties(family='Times New Roman', size=25)
kai_font_prop = FontProperties(family='KaiTi', size=27)
son_font_prop = FontProperties(family='SimSun', size=23)

# 青色，橙色，砖红色，蓝色，紫色，灰色，黑色
color_set = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2', '#F3D266', '#E7EFFA', '#999999']
color = ['#FA7F6F', '#BEB8DC', '#82B0D2', '#FFBE7A', '#8ECFC9', '#999999']
# 设置纹理样式
hatch_patterns = ['/', '\\', '//', '-', 'x', '+', '*', '.', '|']
# 示例数据
data = np.array([[1, 5, 9],  # 第一组数据
                 [2, 6, 10],  # 第二组数据
                 [3, 7, 11],  # 第二组数据
                 [4, 8, 12],  # 第二组数据
                 [5, 9, 13],  # 第二组数据
                 [6, 10, 14],  # 第三组数据
                 [7, 11, 15],  # 第三组数据
                 [9, 13, 17]])  # 第四组数据

groups_data = get_solutions_info(['exp0', 'exp2', 'b_exp'])
for i, group_data in enumerate(groups_data):
    for j, key in enumerate(group_data):
        data[j, i] = group_data[key][0]
        if key == 'BC' and i == 2:
            data[j, i] = 0

# 绘制分组柱状图
# x = np.arange(3)  # x 轴坐标
x = np.array([0, 1, 2])  # x 轴坐标
width = 0.1  # 柱状图宽度
labels = ['实验组$1$', '实验组$2$', '实验组$3$']  # x 轴标签
algorithm = ['DDPG', 'BC-DDPG', 'TD3', 'BC', 'LR-Instant', 'JSPRR', 'Greedy', 'Cloud']

fig, ax = plt.subplots(figsize=(16, 9))
for i, _ in enumerate(algorithm):
    plt.bar(x + i * width, data[i], width,
            alpha=0.8,
            color=color_set[i],
            label=algorithm[i],
            # hatch=hatch_patterns[i],
            edgecolor='black'
            )

plt.ylabel('总时延(毫秒)', font=kai_font_prop)
plt.xticks(x + 4 * width, labels, fontproperties=kai_font_prop)
plt.yticks(fontproperties=roman_font_prop)

ax.xaxis.set_minor_locator(MultipleLocator(0.5))
# ax.yaxis.set_minor_locator(MultipleLocator(2))
# ax.grid(True, which='both', linestyle='--', linewidth=1.5, color='gray', alpha=0.7)

style = {
    'linestyle': '--',
    'linewidth': 1.5,
    'color': 'gray',
    'alpha': 0.2
}
# 设置网格
ax.yaxis.grid(True, which='major', **style)  # x坐标轴的网格使用主刻度
ax.xaxis.grid(True, which='minor', **style)  # y坐标轴的网格使用次刻度

ax.legend(prop=roman_font_prop, ncol=3, loc='upper left')
# 隐藏x轴刻度线和刻度值
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
plt.tight_layout()

# 显示图形
plt.show()
