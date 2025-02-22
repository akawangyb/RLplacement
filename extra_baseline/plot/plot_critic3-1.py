# --------------------------------------------------
# 文件名: plot_critic3-1
# 创建时间: 2025/2/21 23:46
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import os.path

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec

from tools import read_data, sliding_average, read_critic, find_name_log_folders

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


def draw_imitation_fig1(ax, title, file_path=r'2exp_3',
                        annotate_text=r'$\times10^{11}$',
                        critic=False):
    episodes, total_rewards = read_data(file_path) if not critic else read_critic(file_path)
    print(file_path)
    print(total_rewards)
    # total_rewards = sliding_average(total_rewards, 5)
    # 绘制total reward的折线图
    ax.plot(episodes, total_rewards,
            linewidth=3.0,
            color=color[2], )
    ax.set_xlim(0, 200)
    # ax.set_ylim(0, 200)
    plt.xticks(fontproperties=roman_font_prop)
    plt.yticks(fontproperties=roman_font_prop)
    # 添加数量级表示
    ax.annotate(annotate_text, xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
    ax.set_xlabel('学习步数（个）', fontproperties=kai_font_prop)
    ax.set_ylabel('损失函数值', fontproperties=kai_font_prop)
    ax.set_title(title, y=-0.33, fontproperties=kai_font_prop, size=25)
    ax.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.7)


def draw_imitation_fig2(ax, file_paths, title='$(\mathrm{b})$不同模仿学习回合下的后续训练收益',
                        annotate_text=r'$\times 10^6$', tag='lmbda'):
    config_paths = [os.path.join(ele, 'train_config.yaml') for ele in file_paths]
    label = []
    for ele in config_paths:
        with open(ele, 'r', encoding='utf-8') as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)
            print(yaml_data[tag])
        label.append(f"Step=" + str(yaml_data[tag]))
    file_paths = [os.path.join(ele, 'output_info.log') for ele in file_paths]
    for i, file_path in enumerate(file_paths):
        print(file_path)
        episodes, total_rewards = read_data(file_path)

        total_rewards = sliding_average(total_rewards, 10)
        # 绘制total reward的折线图
        ax.plot(episodes[:epoch],
                total_rewards[:epoch],
                label=label[i],
                linewidth=3.0,
                color=color[i])
    ax.set_xlim(0, epoch)
    plt.xticks(fontproperties=roman_font_prop)
    plt.yticks(fontproperties=roman_font_prop)
    # 添加数量级表示
    ax.annotate(annotate_text, xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
    ax.set_xlabel('训练轮次（回合）', fontproperties=kai_font_prop)
    ax.set_ylabel('累计奖励$(\mathrm{ms})$', fontproperties=kai_font_prop)
    # ax.set_ylabel('损失函数值$', fontproperties=kai_font_prop)
    ax.set_title(title, y=-0.33, fontproperties=kai_font_prop, size=25)
    ax.legend(prop=roman_font_prop, loc='lower right', ncol=1)
    ax.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.7)


def draw_sub_fig(ax, file_path, title='$(\mathrm{Step=0})$',
                 annotate_text=r'$\times 10^6$', i=0, tag='lmbda'):
    # config_paths = [os.path.join(ele, 'train_config.yaml') for ele in file_paths]
    # label = []
    # for ele in config_paths:
    #     with open(ele, 'r', encoding='utf-8') as file:
    #         yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    #         print(yaml_data[tag])
    #     label.append(f"Step=" + str(yaml_data[tag]))
    # file_paths = [os.path.join(ele, 'output_info.log') for ele in file_paths]
    # for i, file_path in enumerate(file_paths):
    #     print(file_path)
    episodes, total_rewards = read_data(file_path)

    total_rewards = sliding_average(total_rewards, 1)
    # 绘制total reward的折线图
    ax.plot(episodes[:epoch],
            total_rewards[:epoch],
            # label=label[i],
            linewidth=3.0,
            color=color[i])
    ax.set_xlim(50, 300)
    ax.set_ylim(-3.3, -2.7)
    plt.xticks(fontproperties=roman_font_prop)
    plt.yticks(fontproperties=roman_font_prop)
    # 添加数量级表示
    ax.annotate(annotate_text, xy=(0, 1.01), xycoords='axes fraction', fontproperties=roman_font_prop)
    ax.set_title(title, y=-0.96, fontproperties=kai_font_prop, size=25)
    ax.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.7)


title = ['$(\mathrm{a})$评价网络学习的损失值变化',
         '$(\mathrm{b})$不同学习步数下的后续训练收益']

path = [r'../log_res/ablation/3exp1_critic/critic_loss.log',
        r'../log_res/ablation/3exp1_critic']

# 生成示例数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = 0.9 * np.sin(x) + 0.1
y3 = 1.1 * np.sin(x) - 0.1
y4 = np.cos(x)
y5 = 0.8 * np.cos(x) + 0.2
y6 = np.tan(x)
y7 = 0.7 * np.tan(x) + 0.3

# 创建一个2x4的gridspec布局
gs = GridSpec(4, 2)

# 创建图形
fig = plt.figure(figsize=(16, 10))

# 左边的图占用左边的4行
ax1 = fig.add_subplot(gs[:2, 0])
draw_imitation_fig1(ax1, title[0], path[0], critic=True)

ax2 = fig.add_subplot(gs[2:, 0])
file_paths = find_name_log_folders(path[1], agent_name='imitation_learning')
draw_imitation_fig2(ax2, file_paths, title=title[1], tag='lmbda')

index = 0
# 右边上面的第一个图
ax3 = fig.add_subplot(gs[0, 1])
path = r'../log_res/ablation/3exp1_critic/imitation_learning_env-20240814-035948/output_info.log'
draw_sub_fig(ax3, title='$\mathrm{(c)Step=0}$的后续训练收益', file_path=path, i=index)
index += 1
# 右边上面的第二个图
ax4 = fig.add_subplot(gs[1, 1])
path = r'../log_res/ablation/3exp1_critic/imitation_learning_env-20240814-044156/output_info.log'
draw_sub_fig(ax4, title='$\mathrm{(d)Step=30}$的后续训练收益', file_path=path, i=index)
index += 1
# 右边下面的第一个图
ax5 = fig.add_subplot(gs[2, 1])
path = r'../log_res/ablation/3exp1_critic/imitation_learning_env-20240814-054537/output_info.log'
draw_sub_fig(ax5, title='$\mathrm{(e)Step=60}$的后续训练收益', file_path=path, i=index)
index += 1
# 右边下面的第二个图
ax6 = fig.add_subplot(gs[3, 1])
path = r'../log_res/ablation/3exp1_critic/imitation_learning_env-20240815-015656/output_info.log'
draw_sub_fig(ax6, title='$\mathrm{(f)Step=100}$的后续训练收益', file_path=path, i=index)

# 调整子图间距
plt.tight_layout()
fig.subplots_adjust(left=0.07, right=0.95, bottom=0.12, top=0.95, wspace=0.09, hspace=1.5)

# 显示图形
plt.show()
