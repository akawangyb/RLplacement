# --------------------------------------------------
# 文件名: plot
# 创建时间: 2024/7/13 14:46
# 描述: 选择4种曲线，1Docker，2上海基站，3Uber City,4Uber Airport
# 作者: WangYuanbo
# --------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

roman_font_prop = FontProperties(family='Times New Roman', size=25)
kai_font_prop = FontProperties(family='KaiTi', size=27)
son_font_prop = FontProperties(family='SimSun', size=23)
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'DejaVu Serif'
plt.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
plt.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'
# 设置全局线条粗细
# plt.rcParams['lines.linewidth'] = 2.0

color = ['#FA7F6F', '#BEB8DC', '#82B0D2', '#FFBE7A', '#8ECFC9', '#999999']
marker = ['o', 'D', '^', 'v', '*']
i = 0

fig, ax = plt.subplots(figsize=(16, 9))
plt.xticks(fontproperties=roman_font_prop)
plt.yticks(fontproperties=roman_font_prop)
# ax.set_linewidth(1.5)

# 读取Uber数据
df = pd.read_csv('uber/data.csv')
# 根据地点分别筛选数据
df_airport = df[df['Point'] == 'Airport']
df_city = df[df['Point'] == 'City']

# 使用MinMaxScaler归一化数据
airport_max = df_airport['Count'].max()
city_max = df_city['Count'].max()
df_airport['Count_normalized'] = df_airport['Count'] / airport_max
df_city['Count_normalized'] = df_city['Count'] / city_max

# 在同一个坐标系中绘制两个地点的归一化曲线
ax.plot(df_airport['Hour'],
        df_airport['Count_normalized'],
        label='机场打车服务',
        color=color[i],
        marker=marker[i],
        markersize=12,
        linewidth=3.0)
plt.fill_between(df_airport['Hour'],
                 np.clip(df_airport['Count_normalized'] - 0.1, 0, None),
                 np.clip(df_airport['Count_normalized'] + 0.1, None, 1),
                 alpha=0.2,
                 color=color[i])
i += 1

ax.plot(df_city['Hour'],
        df_city['Count_normalized'],
        label='市内打车服务',
        color=color[i],
        marker=marker[i],
        markersize=12,
        linewidth=3.0)
plt.fill_between(df_city['Hour'],
                 np.clip(df_city['Count_normalized'] - 0.1, 0, None),
                 np.clip(df_city['Count_normalized'] + 0.1, None, 1),
                 alpha=0.2,
                 color=color[i])
i += 1

# 显示基站数据
df = pd.read_csv('base_station/output.csv')
# 将日期和小时列转换为日期时间类型
df['Date'] = pd.to_datetime(df['Date'])
df['Hour'] = pd.to_numeric(df['Hour'])
grouped = df.groupby('Hour')['Count'].agg(['mean', 'std'])
mean_values = grouped['mean']
std_values = grouped['std']
hours = grouped.index
lower_limit = mean_values - std_values
lower_limit = np.clip(lower_limit, 0, None)
upper_limit = mean_values + std_values
max_value = upper_limit.max()
normalized_upper = upper_limit / max_value
normalized_lower = lower_limit / max_value
normalized_mean = mean_values / max_value
ax.plot(hours,
        normalized_mean,
        label='电信基站服务',
        color=color[i],
        marker=marker[i],
        markersize=12,
        linewidth=3.0)
plt.fill_between(hours, normalized_lower, normalized_upper, alpha=0.2 ,color=color[i])
i += 1

# 显示IBMDocker
df = pd.read_csv('ibm_docker/count.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Hour'] = pd.to_numeric(df['Hour'])
grouped = df.groupby('Hour')['Count'].agg(['mean', 'std'])
mean_values = grouped['mean']
std_values = grouped['std']
hours = grouped.index
lower_limit = mean_values - std_values
lower_limit = np.clip(lower_limit, 0, None)
upper_limit = mean_values + std_values
max_value = upper_limit.max()
normalized_upper = upper_limit / max_value
normalized_lower = lower_limit / max_value
normalized_mean = mean_values / max_value

# 绘制折线图
ax.plot(hours,
        normalized_mean,
        label='镜像仓库服务',
        color=color[i],
        marker=marker[i],
        markersize=12,
        linewidth=3.0)
plt.fill_between(hours, normalized_lower, normalized_upper, alpha=0.2,color=color[i])

# 设置标题和轴标签
plt.xticks(range(len(hours)))

ax.set_xlim(left=0, right=23)
ax.set_ylim(0, 1.15)
plt.xticks(fontproperties=roman_font_prop)
plt.yticks(fontproperties=roman_font_prop)
# 添加数量级表示

ax.set_xlabel('时隙', fontproperties=kai_font_prop)
ax.set_ylabel('请求数归一化比例', fontproperties=kai_font_prop)

ax.legend(prop=son_font_prop, loc='upper left', ncol=4)

# 调整布局以减少上方的空白
plt.subplots_adjust(top=0.97, bottom=0.10, left=0.08, right=0.95)

# 灰色的虚线网格，线条粗细为 0.5
ax.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.2)

# plt.savefig('tmp.png')
# 显示图表
plt.show()
