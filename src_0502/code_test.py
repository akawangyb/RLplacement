# --------------------------------------------------
# 文件名: code_test
# 创建时间: 2024/5/5 21:30
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 定义均值和标准差
mu, sigma = 0, 0.5

# 从正态分布中随机抽取1000个样本
s = np.random.normal(mu, sigma, 1000)

# 创建直方图
count, bins, ignored = plt.hist(s, 30, density=True)

# 画出理论的正态分布曲线
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')

plt.show()