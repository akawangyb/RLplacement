# --------------------------------------------------
# 文件名: reward_a_bar
# 创建时间: 2024/7/22 23:56
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec

from tools import get_solutions_info

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
hatch_style = ['/', '\\', 'x', '.', 'o', '+', '-', '|']


def get_dict_data(exp_name='exp0'):
    # 假设有一个字典数据
    data = get_solutions_info([exp_name])
    data = data[0]
    # 提取字典键和值
    labels = []
    values = []
    for key, value in data.items():
        if key in ['DDPG', 'TD3', 'BC']:
            continue
        labels.append(key)
        values.append(value[0] / 1e6)
        print(key, value[0])
    return labels, values


fig = plt.figure(figsize=(16, 15))
# 使用GridSpec对象设置子图的布局
gs = GridSpec(3, 1, hspace=0.35)

# 第一个组实验
ax1 = fig.add_subplot(gs[0, 0])
# 创建柱状图
labels, values = get_dict_data('exp0')
bars = ax1.bar(labels, values, color='white', edgecolor=color, alpha=0.5)
for i, bar in enumerate(bars):
    bar.set_hatch(hatch_style[i])
    bar.set_linewidth(3.0)
for i in range(len(labels)):
    ax1.text(labels[i], values[i], str(int(values[i] * 1e6)), ha='center', va='bottom', fontsize=23)
# 添加标签和标题
ax1.annotate(r'$\times 10^6$', xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
ax1.set_xlabel('实验一', fontproperties=kai_font_prop)
ax1.set_ylabel('总时延', fontproperties=kai_font_prop)
plt.xticks(labels, fontproperties=roman_font_prop, size=20)
plt.yticks(fontproperties=roman_font_prop, size=20)




# 第2个组实验
ax2 = fig.add_subplot(gs[1, 0])
# 创建柱状图
labels, values = get_dict_data('exp2')
bars = ax2.bar(labels, values, color='white', edgecolor=color, alpha=0.5)
for i, bar in enumerate(bars):
    bar.set_hatch(hatch_style[i])
    bar.set_linewidth(3.0)
for i in range(len(labels)):
    ax2.text(labels[i], values[i], str(int(values[i] * 1e6)), ha='center', va='bottom', fontsize=23)
# 添加标签和标题
ax2.annotate(r'$\times 10^6$', xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
ax2.set_xlabel('实验二', fontproperties=kai_font_prop)
ax2.set_ylabel('总时延', fontproperties=kai_font_prop)
plt.xticks(labels, fontproperties=roman_font_prop, size=20)
plt.yticks(fontproperties=roman_font_prop, size=20)

# 第3个组实验
ax3 = fig.add_subplot(gs[2, 0])
# 创建柱状图
labels, values = get_dict_data('b_exp')
bars = ax3.bar(labels, values, color='white', edgecolor=color, alpha=0.5)
for i, bar in enumerate(bars):
    bar.set_hatch(hatch_style[i])
    bar.set_linewidth(3.0)
for i in range(len(labels)):
    ax3.text(labels[i], values[i], str(int(values[i] * 1e6)), ha='center', va='bottom', fontsize=23)
# 添加标签和标题
ax3.annotate(r'$\times 10^6$', xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
ax3.set_xlabel('实验三', fontproperties=kai_font_prop)
ax3.set_ylabel('总时延', fontproperties=kai_font_prop)
plt.xticks(labels, fontproperties=roman_font_prop, size=20)
plt.yticks(fontproperties=roman_font_prop, size=20)

plt.subplots_adjust(top=0.97, bottom=0.08, left=0.07, right=0.97)

# 隐藏x轴刻度线和刻度值
# plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
# 隐藏x轴刻度线和刻度值
# plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=True)

plt.show()
