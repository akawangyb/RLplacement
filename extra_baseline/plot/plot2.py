# --------------------------------------------------
# 文件名: plot2
# 创建时间: 2024/8/11 22:05
# 描述: 生成实验的第二个图
# 作者: WangYuanbo
# --------------------------------------------------
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec

# from epoch_interference_compare import get_reward_data
from tools import get_dict_data, get_delay_data

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

# 创建一个Figure对象，并设置图像大小
fig = plt.figure(figsize=(16, 15))

# 使用GridSpec对象设置子图的布局
gs = GridSpec(3, 2, hspace=0.35, wspace=0.15)


# 数据格式
# -算法
#   --[total_reward, episode_reward, episode_interference, episode_time]

def draw_interference_fig(ax, exp_name, title, top=-1, edge_server_number=0):
    labels, values = get_dict_data(exp_name)
    print(values[0])
    for i in range(4):
        edge_service_number = 0
        for time_ele in values[i]:
            edge_service_number = 0
            for ele in time_ele:
                if ele != 0:
                    edge_service_number += 1
        print(values[0], "edge_service_number", edge_service_number)
        # ax.plot(range(24), [sum(ele) / len(ele) for ele in values[i]],
        # ax.plot(range(24), [sum(ele) for ele in values[i]],
        # ax.plot(range(24), [sum(ele)/edge_service_number for ele in values[i]],
        ax.plot(range(24), [sum(ele) / edge_server_number for ele in values[i]],
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

    legend_font = FontProperties(family='Times New Roman', size=15)
    ax.legend(prop=legend_font, loc='upper right', ncol=4)


def draw_delay_fig(ax, exp_name, title, top=-1, edge_server_number=0):
    labels, values = get_delay_data(exp_name)
    print(values[0])
    for i in range(4):
        # ax.plot(range(24), [sum(ele) / (1e3 * len(ele)) for ele in values[i]],
        ax.plot(range(24), [sum(ele) / (1e3 * edge_server_number) for ele in values[i]],
                label=labels[i], linewidth=3.0, markersize=10,
                color=color[i], marker=marker[i])
    ax.set_xlim(0, 23)
    if top != -1:
        ax.set_ylim(top=top)
    plt.xticks(range(24), fontproperties=roman_font_prop, size=20)
    plt.yticks(fontproperties=roman_font_prop, size=20)
    ax.annotate(r'$\times 10^3$', xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
    ax.set_xlabel('时隙', fontproperties=kai_font_prop)
    ax.set_ylabel('平均延迟$(\mathrm{ms})$', fontproperties=kai_font_prop)
    ax.set_title(title, y=-0.30, fontproperties=kai_font_prop, size=25)
    # 灰色的虚线网格，线条粗细为 0.5
    ax.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.7)

    legend_font = FontProperties(family='Times New Roman', size=15)
    ax.legend(prop=legend_font, loc='upper right', ncol=4)


# def draw_time_fig(ax, exp_name, title, top=-1):
#     labels, values = get_time_data(exp_name)
#     # assert 24 == len(values),f'values len is {len(values)}'
#     # values = sum(values) / len(values)
#     bars = ax.bar(labels, values, color='white', edgecolor=color, alpha=0.5)
#     for i, bar in enumerate(bars):
#         bar.set_hatch(hatch_style[i])
#         bar.set_linewidth(3.0)
#     for i in range(len(labels)):
#         ax.text(labels[i], values[i], str(int(values[i] * 1e6)), ha='center', va='bottom',
#                 fontsize=20)
#
#     if top != -1:
#         ax.set_ylim(top=top)
#     # 添加标签和标题
#     ax.annotate(r'$\times 10^3$', xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
#     # ax.set_xlabel(x_name, fontproperties=kai_font_prop)
#     ax.set_title(title, y=-0.30, fontproperties=kai_font_prop, size=25)
#     ax.set_ylabel('总时延', fontproperties=kai_font_prop)
#     plt.xticks(labels, fontproperties=roman_font_prop, size=20)
#     plt.yticks(fontproperties=roman_font_prop, size=20)


std_para = [-1, -1, -1, -1, -1, -1]
exp1_para = [0.32, 0.35, 2.1, 3.3, 5.4, 16]
exp2_para = [0.35, 0.27, 0.24, 4.5, 6.3, 4.0]
exp3_para = [0.59, 0.59, 0.93, 5.2, 5.8, 5.5]
edge_server_number = [40, 40, 40]
para = exp3_para
# para = std_para

index = 0

# 第4个组实验


exp_list = ['3exp_1', '3exp_2', '3exp_3', '3exp_1', '3exp_2', '3exp_3']
exp1_title_list = [
    '$(\mathrm{a})$ 请求数量设置为$30$的每时隙干扰因子对比',
    '$(\mathrm{b})$ 请求数量设置为$50$的每时隙干扰因子对比',
    '$(\mathrm{c})$ 请求数量设置为$120$的每时隙干扰因子对比',
    '$(\mathrm{d})$ 请求数量设置为$30$的每时隙平均时延对比',
    '$(\mathrm{e})$ 请求数量设置为$50$的每时隙平均时延对比',
    '$(\mathrm{f})$ 请求数量设置为$120$的每时隙平均时延对比',

]
exp2_title_list = [
    '$(\mathrm{a})$机场打车服务分布采样的每时隙干扰因子对比',
    '$(\mathrm{b})$基站通信服务分布采样的每时隙干扰因子对比',
    '$(\mathrm{c})$容器镜像服务分布采样的每时隙干扰因子对比',
    '$(\mathrm{d})$机场打车服务分布采样的每时隙平均时延对比',
    '$(\mathrm{e})$基站通信服务分布采样的每时隙平均时延对比',
    '$(\mathrm{f})$容器镜像服务分布采样的每时隙平均时延对比',
]

exp3_title_list = [
    '$(\mathrm{a})$ 混合请求类型一的每时隙干扰因子对比',
    '$(\mathrm{b})$ 混合请求类型二的每时隙干扰因子对比',
    '$(\mathrm{c})$ 混合请求类型三的每时隙干扰因子对比',
    '$(\mathrm{d})$ 混合请求类型一的每时隙平均时延对比',
    '$(\mathrm{e})$ 混合请求类型二的每时隙平均时延对比',
    '$(\mathrm{f})$ 混合请求类型三的每时隙平均时延对比',
]

exp4_title_list = [
    '$(\mathrm{a})$ 服务器设置为$5$的每时隙干扰因子对比',
    '$(\mathrm{b})$ 服务器设置为$7$的每时隙干扰因子对比',
    '$(\mathrm{c})$ 服务器设置为$9$的每时隙干扰因子对比',
    '$(\mathrm{d})$ 服务器设置为$5$的每时隙平均时延对比',
    '$(\mathrm{e})$ 服务器设置为$7$的每时隙平均时延对比',
    '$(\mathrm{f})$ 服务器设置为$9$的每时隙平均时延对比',
]


title_list = exp3_title_list

ax1 = fig.add_subplot(gs[0, 0])
draw_interference_fig(ax1, exp_list[index], title_list[index], para[index], edge_server_number[0])
index += 1

ax2 = fig.add_subplot(gs[1, 0])
draw_interference_fig(ax2, exp_list[index], title_list[index], para[index], edge_server_number[1])
index += 1

ax3 = fig.add_subplot(gs[2, 0])
draw_interference_fig(ax3, exp_list[index], title_list[index], para[index], edge_server_number[2])
index += 1

ax4 = fig.add_subplot(gs[0, 1])
draw_delay_fig(ax4, exp_list[index], title_list[index], para[index], edge_server_number[0])
index += 1

ax5 = fig.add_subplot(gs[1, 1])
draw_delay_fig(ax5, exp_list[index], title_list[index], para[index], edge_server_number[1])
index += 1

ax6 = fig.add_subplot(gs[2, 1])
draw_delay_fig(ax6, exp_list[index], title_list[index], para[index], edge_server_number[2])
index += 1

plt.subplots_adjust(top=0.98, bottom=0.08, left=0.068, right=0.99)
plt.show()
