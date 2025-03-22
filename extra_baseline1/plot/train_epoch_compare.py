# --------------------------------------------------
# 文件名: exp_result_compare
# 创建时间: 2024/7/17 14:38
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from tools import find_log_folders
from tools import read_data, sliding_average
from train_epoch_compare import draw_a_subfig


def ax2in(ax, file_paths):
    # 创建放大的局部子图
    axins = inset_axes(ax, width="40%", height="50%", loc='lower left',
                       bbox_to_anchor=(0.18, 0.22, 2.0, 1),
                       bbox_transform=ax.transAxes,
                       )
    for i, file_path in enumerate(file_paths):
        episodes, total_rewards = read_data(file_path)
        total_rewards = sliding_average(total_rewards, 5)
        # 绘制total reward的折线图
        axins.plot(episodes, total_rewards, linewidth=3.0, color=color[i])

    #  框的范围
    xlim0 = 0
    xlim1 = 1000
    ylim0 = -6.3
    ylim1 = -5.6

    # ylim0 = -0.5
    # ylim1 = -1.5
    # 调整子坐标系的显示范围
    sub_ax_xlim0 = xlim0
    sub_ax_xlim1 = xlim1
    sub_ax_ylim0 = ylim0
    sub_ax_ylim1 = ylim1
    axins.set_xlim(sub_ax_xlim0, sub_ax_xlim1)
    axins.set_ylim(sub_ax_ylim0, sub_ax_ylim1)

    # 原图中画方框
    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, linewidth=2.0, color='black', linestyle='--')

    # 画两条线
    xy = (xlim0, ylim0)
    xy2 = (xlim0, sub_ax_ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax, alpha=0.8)
    con.set_linestyle('--')
    con.set_linewidth(1.5)
    axins.add_artist(con)

    xy = (xlim1, ylim0)
    xy2 = (xlim1, sub_ax_ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax, alpha=0.8)
    # 设置连接线的粗细为2.0
    con.set_linewidth(1.5)
    # 设置连接线为虚线
    con.set_linestyle('--')
    axins.add_artist(con)
    axins.set_xlim(xlim0, xlim1)
    axins.annotate(r'$\times 10^6$', xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop, size=15)
    axins.grid(True, linestyle='--', linewidth=1.0, color='gray')
    plt.xticks(fontproperties=roman_font_prop, size=15)
    plt.yticks(fontproperties=roman_font_prop, size=15)


# # 设置全局字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题
# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = 'DejaVu Serif'
# plt.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
# plt.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'

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

# 创建一个Figure对象，并设置图像大小
fig = plt.figure(figsize=(16, 15))

# 使用GridSpec对象设置子图的布局
gs = GridSpec(3, 1, hspace=0.35)

file_paths1 = find_log_folders(r'../log_res/1exp_1')
file_paths2 = find_log_folders(r'../log_res/1exp_2')
file_paths3 = find_log_folders(r'../log_res/1exp_3')
# file_paths1 = find_log_folders(r'../log_res/exp1_1')
# file_paths2 = find_log_folders(r'../log_res/exp1_2')
# file_paths3 = find_log_folders(r'../log_res/exp1_3')
file_paths1 = [ele + '/output_info.log' for ele in file_paths1]
file_paths2 = [ele + '/output_info.log' for ele in file_paths2]
file_paths3 = [ele + '/output_info.log' for ele in file_paths3]
print(file_paths2)
label = ['DDPG', 'BC-DDPG', 'TD3', ]
epoch = 1000


# 创建第一个子图

def draw_a_subfig(ax, file_paths, title='$(\mathrm{a})$ 实验组$1$', annotate_text=r'$\times 10^6$', fator=False):
    for i, file_path in enumerate(file_paths):
        episodes, total_rewards = read_data(file_path)
        if fator:
            total_rewards = [ele / 10 for ele in total_rewards]
        total_rewards = sliding_average(total_rewards, 5)
        # 绘制total reward的折线图
        ax.plot(episodes[:epoch], total_rewards[:epoch], label=label[i], linewidth=3.0, color=color[i])
    ax.set_xlim(0, epoch)
    plt.xticks(fontproperties=roman_font_prop)
    plt.yticks(fontproperties=roman_font_prop)
    # 添加数量级表示
    ax.annotate(annotate_text, xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
    ax.set_xlabel('训练轮次（回合）', fontproperties=kai_font_prop)
    ax.set_ylabel('累计奖励（毫秒）', fontproperties=kai_font_prop)
    ax.set_title(title, y=-0.30, fontproperties=kai_font_prop, size=25)
    ax.legend(prop=roman_font_prop, loc='lower right', ncol=4)
    ax.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.7)


ax1 = fig.add_subplot(gs[0, 0])
draw_a_subfig(ax1, file_paths1, '$(\mathrm{a})$ 实验组$1$', r'$\times 10^6$')

# ax2in(ax1, file_paths1)

# 创建第二个子图
ax2 = fig.add_subplot(gs[1, 0])
draw_a_subfig(ax2, file_paths2, '$(\mathrm{b})$ 实验组$2$', r'$\times 10^6$')

# ax2in(ax2, file_paths2)

# 创建第三个子图
ax3 = fig.add_subplot(gs[2, 0])
draw_a_subfig(ax3, file_paths3, '$(\mathrm{c})$ 实验组$3$', r'$\times 10^7$',True)

plt.xticks(fontproperties=roman_font_prop)
plt.yticks(fontproperties=roman_font_prop)

plt.subplots_adjust(top=0.98, bottom=0.08, left=0.07, right=0.97)

# 显示图像
plt.show()
