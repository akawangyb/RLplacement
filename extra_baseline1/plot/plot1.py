# --------------------------------------------------
# 文件名: plot1
# 创建时间: 2024/8/11 21:35
# 描述: 生成实验结果的第一个图
# 作者: WangYuanbo
# --------------------------------------------------
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec

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

file_paths1 = find_log_folders(r'../log_res/4exp_1')
file_paths2 = find_log_folders(r'../log_res/4exp_2')
file_paths3 = find_log_folders(r'../log_res/4exp_3')
# file_paths1 = find_log_folders(r'../log_res/exp1_1')
# file_paths2 = find_log_folders(r'../log_res/exp1_2')
# file_paths3 = find_log_folders(r'../log_res/exp1_3')
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


def draw_train_subfig(ax, file_paths, title='$(\mathrm{a})$ 实验组$1$', annotate_text=r'$\times 10^6$', fator=False):
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
    ax.set_ylabel('累计奖励$(\mathrm{ms})$', fontproperties=kai_font_prop)
    ax.set_title(title, y=-0.30, fontproperties=kai_font_prop, size=25)
    ax.legend(prop=roman_font_prop, loc='lower right', ncol=1)
    ax.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.7)


def draw_totaldelay_fig(ax, exp_name, title, top=-1):
    labels, values = get_reward_data(exp_name)
    print(labels)
    print(values)
    bars = ax.bar(labels, values, color='white', edgecolor=color, alpha=0.5)
    for i, bar in enumerate(bars):
        bar.set_hatch(hatch_style[i])
        bar.set_linewidth(3.0)
    for i in range(len(labels)):
        ax.text(labels[i], values[i], str(int(values[i] * 1e6)), ha='center', va='bottom',
                fontsize=20)

    if top != -1:
        ax.set_ylim(top=top)
    # 添加标签和标题
    ax.annotate(r'$\times 10^6$', xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
    # ax.set_xlabel(x_name, fontproperties=kai_font_prop)
    ax.set_title(title, y=-0.25, fontproperties=kai_font_prop, size=25)
    ax.set_ylabel('总时延$(\mathrm{ms})$', fontproperties=kai_font_prop)
    plt.xticks(labels, fontproperties=roman_font_prop, size=20)
    plt.yticks(fontproperties=roman_font_prop, size=20)


ax1 = fig.add_subplot(gs[0, 0])
# draw_train_subfig(ax1, file_paths1, '$(\mathrm{a})$ 请求数量为$30$的训练奖励对比', r'$\times 10^6$')
exp1_title_list = [
    '$(\mathrm{a})$ 请求数量设置为$30$的训练奖励对比',
    '$(\mathrm{b})$ 请求数量设置为$50$的训练奖励对比',
    '$(\mathrm{c})$ 请求数量设置为$120$的训练奖励对比',
    '$(\mathrm{d})$ 请求数量设置为$30$的总时延对比',
    '$(\mathrm{e})$ 请求数量设置为$50$的总时延对比',
    '$(\mathrm{f})$ 请求数量设置为$120$的总时延对比',
]
exp2_title_list = [
    '$(\mathrm{a})$ 机场打车服务分布采样的训练奖励对比',
    '$(\mathrm{b})$ 基站通信服务分布采样的训练奖励对比',
    '$(\mathrm{c})$ 容器镜像服务分布采样的训练奖励对比',
    '$(\mathrm{d})$ 机场打车服务分布采样的总时延对比',
    '$(\mathrm{e})$ 基站通信服务分布采样的总时延对比',
    '$(\mathrm{f})$ 容器镜像服务分布采样的总时延对比',
]

exp3_title_list = [
    '$(\mathrm{a})$ 混合请求类型一的训练奖励对比',
    '$(\mathrm{b})$ 混合请求类型二的训练奖励对比',
    '$(\mathrm{c})$ 混合请求类型三的训练奖励对比',
    '$(\mathrm{d})$ 混合请求类型一的总时延对比',
    '$(\mathrm{e})$ 混合请求类型二的总时延对比',
    '$(\mathrm{f})$ 混合请求类型三的总时延对比',
]

exp4_title_list = [
    '$(\mathrm{a})$ 服务器设置为$5$的训练奖励对比',
    '$(\mathrm{b})$ 服务器设置为$7$的训练奖励对比',
    '$(\mathrm{c})$ 服务器设置为$9$的训练奖励对比',
    '$(\mathrm{d})$ 服务器设置为$5$的总时延对比',
    '$(\mathrm{e})$ 服务器设置为$7$的总时延对比',
    '$(\mathrm{f})$ 服务器设置为$9$的总时延对比',
]

title_list = exp1_title_list
draw_train_subfig(ax1, file_paths1, title_list[0], r'$\times 10^6$', False)
# ax2in(ax1, file_paths1)

# 创建第二个子图
ax2 = fig.add_subplot(gs[1, 0])
draw_train_subfig(ax2, file_paths2, title_list[1], r'$\times 10^6$')

# ax2in(ax2, file_paths2)

# 创建第三个子图
ax3 = fig.add_subplot(gs[2, 0])
draw_train_subfig(ax3, file_paths3, title_list[2], r'$\times 10^6$', False)

index = 0
# exp1_para = [2.9, 4.9, 11.9]
exp1_para = [0.6, 1.0, 1.5]
exp2_para = [3.5, 3.5, 2.4]
exp3_para = [4.7, 4.2, 3.8]
exp4_para = [19, 12, 12]

std_para = [-1, -1, -1]
para = std_para
# para = exp1_para
exp_list = ['4exp_1', '4exp_2', '4exp_3']

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
