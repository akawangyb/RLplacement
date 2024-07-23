# --------------------------------------------------
# 文件名: exp_result_compare
# 创建时间: 2024/7/17 14:38
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import matplotlib.pyplot as plt
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

roman_font_prop = FontProperties(family='Times New Roman', size=18)
kai_font_prop = FontProperties(family='KaiTi', size=22)
son_font_prop = FontProperties(family='SimSun', size=22)

# 青色，橙色，砖红色，蓝色，紫色，灰色，黑色
color_set = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2', '#999999']
color = ['#FA7F6F', '#BEB8DC', '#82B0D2', '#FFBE7A', '#8ECFC9', '#999999']
marker = ['x', 'o', '*', '^', 'v']
hatch_style = ['/', '\\', 'x', '.', 'o', '+', '-', '|']


def get_dict_data(exp_name='exp0'):
    # 假设有一个字典数据
    data = get_solutions_info([exp_name])
    data = data[0]
    # 提取字典键和值
    labels = []
    values = []
    for key, value in data.items():
        if key in ['DDPG', 'TD3', 'BC', 'Cloud']:
            continue
        labels.append(key)
        values.append(value[2])
        print(key, value[0])
    return labels, values


def draw_a_line_fig(ax, exp_name, title, top=-1):
    labels, values = get_dict_data(exp_name)
    print(values[0])
    for i in range(4):
        ax.plot(range(24), [sum(ele) / len(ele) for ele in values[i]],
                label=labels[i], linewidth=3.0, markersize=10,
                color=color[i], marker=marker[i])
    ax.set_xlim(0, 23)
    if top != -1:
        ax.set_ylim(top=top)
    plt.xticks(range(24), fontproperties=roman_font_prop, size=20)
    plt.yticks(fontproperties=roman_font_prop, size=20)

    ax.set_xlabel('时隙', fontproperties=kai_font_prop)
    ax.set_ylabel('平均干扰因子', fontproperties=kai_font_prop)
    ax.set_title(title, y=-0.30, fontproperties=kai_font_prop, size=25)
    # 灰色的虚线网格，线条粗细为 0.5
    ax.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.7)

    ax.legend(prop=roman_font_prop, loc='upper right', ncol=2)


exp_list = ['exp0', 'exp2', 'b_exp', 'exp0', 'exp2', 'b_exp']
title_list = ['$(\mathrm{a})$ 实验组$1$每时隙平均干扰因子',
              '$(\mathrm{b})$ 实验组$2$每时隙平均干扰因子',
              '$(\mathrm{c})$ 实验组$3$每时隙平均干扰因子',
              '$(\mathrm{d})$ 实验组$1$的总时延对比',
              '$(\mathrm{e})$ 实验组$2$的总时延对比',
              '$(\mathrm{f})$ 实验组$3$的总时延对比']

gs = GridSpec(3, 2, hspace=0.35, wspace=0.13)
fig = plt.figure(figsize=(16, 15))
index = 0
ax1 = fig.add_subplot(gs[0, 1])
draw_a_line_fig(ax1, exp_list[index], title_list[index], 0.39)
index += 1

# 第二个
ax2 = fig.add_subplot(gs[1, 1])
draw_a_line_fig(ax2, exp_list[index], title_list[index], 0.3)
index += 1

# 第三个
ax3 = fig.add_subplot(gs[2, 1])
draw_a_line_fig(ax3, exp_list[index], title_list[index], -1)
index += 1


def get_reward_data(exp_name='exp0'):
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


def draw_a_sub_fig(ax, exp_name, title):
    labels, values = get_reward_data(exp_name)
    bars = ax.bar(labels, values, color='white', edgecolor=color, alpha=0.5)
    for i, bar in enumerate(bars):
        bar.set_hatch(hatch_style[i])
        bar.set_linewidth(3.0)
    for i in range(len(labels)):
        ax.text(labels[i], values[i], str(int(values[i] * 1e6)), ha='center', va='bottom',
                fontsize=20)

    # 添加标签和标题
    ax.annotate(r'$\times 10^6$', xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
    # ax.set_xlabel(x_name, fontproperties=kai_font_prop)
    ax.set_title(title, y=-0.30, fontproperties=kai_font_prop, size=25)
    ax.set_ylabel('总时延', fontproperties=kai_font_prop)
    plt.xticks(labels, fontproperties=roman_font_prop, size=20)
    plt.yticks(fontproperties=roman_font_prop, size=20)


# 第4个组实验
ax4 = fig.add_subplot(gs[0, 0])

draw_a_sub_fig(ax4, exp_list[index], title_list[index])
index += 1

# 第5个组实验
ax5 = fig.add_subplot(gs[1, 0])
draw_a_sub_fig(ax5, exp_list[index], title_list[index])
index += 1

# 第6个组实验
ax6 = fig.add_subplot(gs[2, 0])
draw_a_sub_fig(ax6, exp_list[index], title_list[index])
index += 1

plt.subplots_adjust(top=0.98, bottom=0.08, left=0.06, right=0.99)

plt.show()
