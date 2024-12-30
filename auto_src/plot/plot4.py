# --------------------------------------------------
# 文件名: plot_set_epochs
# 创建时间: 2024/8/13 18:39
# 描述: 如何设置模仿学习回合数
# 作者: WangYuanbo
# --------------------------------------------------
# 设置全局字体
import os.path

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec

from tools import find_name_log_folders, read_data, read_critic, sliding_average

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
color = ['#8ECFC9', '#FFBE7A', '#FA7F6F',
         '#82B0D2', '#BEB8DC', '#E7DAD2',
         '#F3D266', '#E7EFFA', '#999999',
         '#F5EBAE', '#EF8B67', '#992224',
         '#8074C8', '#D6EFF4', '#D8B365',
         '#5BB5AC', '#DE526C', '#6F6F6F',
         '#DD7C4F', '#6C61AF', '#B54764',
         '#f2fafc']
marker = ['x', 'o', '*', '^', 'v']
hatch_style = ['/', '\\', 'x', '.', 'o', '+', '-', '|']

epoch = 500


def draw_critic_loss(ax, title, file_path=r'2exp_3',
                     annotate_text=r'$\times 10^{11}$',
                     ):
    episodes, total_rewards = read_critic(file_path)
    ax.plot(episodes, total_rewards,
            linewidth=3.0,
            color=color[2], )
    ax.set_xlim(0, 200)
    plt.xticks(fontproperties=roman_font_prop)
    plt.yticks(fontproperties=roman_font_prop)
    # 添加数量级表示
    ax.annotate(annotate_text, xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
    ax.set_xlabel('更新步数（个）', fontproperties=kai_font_prop)
    ax.set_ylabel('损失函数值', fontproperties=kai_font_prop)
    ax.set_title(title, y=-0.22, fontproperties=kai_font_prop, size=25)
    ax.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.7)


def draw_imitation_fig2(ax, file_paths, title='$(\mathrm{b})$不同评价网络学习回合设置下的后续训练收益',
                        annotate_text=r'$\times 10^6$', ):
    label = [0, 20, 50, 100]

    file_paths = [os.path.join(ele, 'output_info.log') for ele in file_paths]
    for i, file_path in enumerate(file_paths):
        episodes, total_rewards = read_data(file_path)

        total_rewards = sliding_average(total_rewards, 12)
        # 绘制total reward的折线图
        ax.plot(episodes[:epoch],
                total_rewards[:epoch],
                label='Epoch=' + str(label[i]),
                linewidth=3.0,
                color=color[i])
    ax.set_xlim(0, epoch)
    plt.xticks(fontproperties=roman_font_prop)
    plt.yticks(fontproperties=roman_font_prop)
    # 添加数量级表示
    ax.annotate(annotate_text, xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
    ax.set_xlabel('训练轮次（回合）', fontproperties=kai_font_prop)
    ax.set_ylabel('累计奖励$(\mathrm{ms})$', fontproperties=kai_font_prop)
    ax.set_title(title, y=-0.22, fontproperties=kai_font_prop, size=25)
    ax.legend(prop=roman_font_prop, loc='lower right', ncol=1)
    ax.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.7)


title2_1 = ['$(\mathrm{a})$评价网络学习的损失值变化',
            '$(\mathrm{b})$不同学习回合下的后续训练收益']

title2_2 = ['$(\mathrm{a})$动作网络模仿学习0轮',
            '$(\mathrm{b})$动作网络模仿学习92轮']

path2_1 = [r'../log/x3exp_1/critic_learn/critic_loss.log',
           r'../log/x3exp_1/critic_learn/epoch66']

path2_2 = [r'../log/x3exp_1/critic_learn/epoch0_10',
           r'../log/x3exp_1/critic_learn/epoch92buffer']

path3_1 = [r'../log/x3exp_1/critic_learn/critic_loss.log',
           r'../log/x3exp_1/critic_learn/epoch92buffer']

title = title2_1
path = path2_1
fig = plt.figure(figsize=(16, 7))

gs = GridSpec(1, 2, hspace=0.35, wspace=0.16)
ax1 = fig.add_subplot(gs[0, 0])
draw_critic_loss(ax1, title[0], path[0])
# file_paths = find_name_log_folders(path[0], agent_name='imitation_learning')
# draw_imitation_fig2(ax1, file_paths, title[0])
# print(file_paths)

ax2 = fig.add_subplot(gs[0, 1])
file_paths = find_name_log_folders(path[1], agent_name='imitation_learning')
draw_imitation_fig2(ax2, file_paths, title=title[1])

plt.subplots_adjust(top=0.95, bottom=0.18, left=0.06, right=0.98)
plt.show()
