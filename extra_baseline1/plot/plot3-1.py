# --------------------------------------------------
# 文件名: plot1-1
# 创建时间: 2025/2/28 23:01
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

# from epoch_interference_compare import get_reward_data
from tools import find_log_folders, read_data, sliding_average, get_reward_data

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


def draw_train_subfig(ax, file_paths, title='$(\mathrm{a})$ 实验组$1$', annotate_text=r'$\times 10^6$', factor=False,
                      xlim=(0, 1000),
                      ylim=(-0.5, -0.2),
                      anchor_site=(1000, -0.98),
                      height=1.2,
                      width=4.8,
                      show_axins=True):
    for i, file_path in enumerate(file_paths):
        episodes, total_rewards = read_data(file_path)
        if factor:
            total_rewards = [ele / 10 for ele in total_rewards]
        total_rewards = sliding_average(total_rewards, 1)
        # 绘制total reward的折线图
        ax.plot(episodes[:epoch], total_rewards[:epoch], label=label[i], linewidth=3.0, color=color[i])
    ax.set_xlim(0, epoch)
    # 添加数量级表示
    ax.annotate(annotate_text, xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
    ax.set_xlabel('训练轮次（回合）', fontproperties=kai_font_prop)
    ax.set_ylabel('累计奖励$(\mathrm{ms})$', fontproperties=kai_font_prop)
    ax.set_title(title, y=-0.30, fontproperties=kai_font_prop, size=25)
    ax.legend(prop=roman_font_prop, loc='lower right', ncol=1)
    ax.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.7)

    plt.xticks(fontproperties=roman_font_prop)
    plt.yticks(fontproperties=roman_font_prop)

    if not show_axins:
        return
    # 创建局部放大图
    axins = inset_axes(
        ax,
        loc='lower right',
        height=height,
        width=width,
        bbox_to_anchor=anchor_site,  # 矩形区域 (x0, y0, x1, y1),左下角和右上角
        bbox_transform=ax.transData,
    )
    axins.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.7)
    # 设置局部放大图的显示范围
    x1, x2 = xlim
    y1, y2 = ylim
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # 设置局部放大图的坐标轴刻度
    axins.yaxis.get_major_locator().set_params(integer=True)
    # 添加连接线，将主图中的局部区域和局部放大图关联起来
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", lw=2)
    for i, file_path in enumerate(file_paths):
        episodes, total_rewards = read_data(file_path)
        if factor:
            total_rewards = [ele / 10 for ele in total_rewards]
        total_rewards = sliding_average(total_rewards, 1)
        # 绘制total reward的折线图
        axins.plot(episodes[:epoch], total_rewards[:epoch], label=label[i], linewidth=3.0, color=color[i])

        # 添加数量级表示
    axins.annotate(annotate_text, xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)

    plt.xticks(fontproperties=roman_font_prop)
    plt.yticks(fontproperties=roman_font_prop)


def draw_totaldelay_fig(ax, exp_name, title, top=-1):
    labels, values = get_reward_data(exp_name)
    values = [ele * 10 for ele in values]
    print(labels)
    print(values)
    bars = ax.bar(labels, values, color='white', edgecolor=color, alpha=0.5)
    for i, bar in enumerate(bars):
        bar.set_hatch(hatch_style[i])
        bar.set_linewidth(3.0)
    for i in range(len(labels)):
        ax.text(labels[i], values[i], str(int(values[i] * 1e5)), ha='center', va='bottom',
                fontsize=20)

    if top != -1:
        ax.set_ylim(top=top)
    # 添加标签和标题
    ax.annotate(r'$\times 10^5$', xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
    # ax.set_xlabel(x_name, fontproperties=kai_font_prop)
    ax.set_title(title, y=-0.30, fontproperties=kai_font_prop, size=25)
    ax.set_ylabel('总时延$(\mathrm{ms})$', fontproperties=kai_font_prop)
    plt.xticks(labels, fontproperties=roman_font_prop, size=20)
    plt.yticks(fontproperties=roman_font_prop, size=20)


# draw_train_subfig(ax1, file_paths1, '$(\mathrm{a})$ 请求数量为$30$的训练奖励对比', r'$\times 10^6$')
exp1_title_list = [
    '$(\mathrm{a})$ 混合请求类型一的训练奖励对比',
    '$(\mathrm{b})$ 混合请求类型二的训练奖励对比',
    '$(\mathrm{c})$ 混合请求类型三的训练奖励对比',
    '$(\mathrm{d})$ 混合请求类型一的总时延对比',
    '$(\mathrm{e})$ 混合请求类型二的总时延对比',
    '$(\mathrm{f})$ 混合请求类型三的总时延对比',
]
exp_list = ['6exp_1', '6exp_2', '6exp_3']

file_paths1 = find_log_folders(r'../log_res/6exp_1')
file_paths2 = find_log_folders(r'../log_res/6exp_2')
file_paths3 = find_log_folders(r'../log_res/6exp_3')
file_paths1 = [ele + '/output_info.log' for ele in file_paths1]
file_paths2 = [ele + '/output_info.log' for ele in file_paths2]
file_paths3 = [ele + '/output_info.log' for ele in file_paths3]
print(file_paths2)
label = ['DDPG', 'BC-DDPG', 'TD3']
epoch = 1000

# 创建一个Figure对象，并设置图像大小
fig = plt.figure(figsize=(16, 15))

# 使用GridSpec对象设置子图的布局
gs = GridSpec(3, 2, hspace=0.35, wspace=0.15)
title_list = exp1_title_list

# 创建左边第一个子图
ax1 = fig.add_subplot(gs[0, 0])
draw_train_subfig(ax1, file_paths1, title_list[0], r'$\times 10^6$', False,
                  xlim=(200, 600),
                  ylim=(-0.5, -0.2),
                  anchor_site=(980, -1.50),
                  show_axins=False)

# 创建左边第二个子图
ax2 = fig.add_subplot(gs[1, 0])
draw_train_subfig(ax2, file_paths2, title_list[1], r'$\times 10^6$',
                  xlim=(200, 600),
                  ylim=(-1.0, -0.30),
                  anchor_site=(980, -1.65),
                  show_axins=False)

# 创建左边第三个子图
ax3 = fig.add_subplot(gs[2, 0])
draw_train_subfig(ax3, file_paths3, title_list[2], r'$\times 10^6$', False,
                  xlim=(200, 600),
                  ylim=(-0.5, -0.35),
                  anchor_site=(980, -1.2),
                  height=1.0,
                  width=4.2,
                  show_axins=False)

index = 0
exp1_para = [12, 12, 10]

std_para = [-1, -1, -1]
para = exp1_para
# para = std_para

# 第4个图
ax4 = fig.add_subplot(gs[0, 1])
draw_totaldelay_fig(ax4, exp_list[index], title_list[3], para[index])

index += 1

# 第5个图
ax5 = fig.add_subplot(gs[1, 1])
draw_totaldelay_fig(ax5, exp_list[index], title_list[4], para[index])
index += 1

# 第6个图
ax6 = fig.add_subplot(gs[2, 1])
draw_totaldelay_fig(ax6, exp_list[index], title_list[5], para[index])
index += 1

plt.subplots_adjust(top=0.98, bottom=0.08, left=0.06, right=0.99)
plt.show()
