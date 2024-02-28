# --------------------------------------------------
# 文件名: pso_demo
# 创建时间: 2024/2/24 13:50
# 描述: 粒子群优化算法demo
# 作者: WangYuanbo
# --------------------------------------------------
import numpy as np
# 粒子群算法原理
# 思路一群人一起挖宝，一个人挖到了宝物，大家往他的方向靠近，但是每个人人保留自己的挖宝方向
# 粒子数量
n_particles = 20
# 粒子位置和速度的上下界
bounds = [-10, 10]
# 常量
w = 0.7
c1 = 0.8
c2 = 0.9
# 设置最大迭代次数
n_iterations = 30

# 初始化粒子位置和速度
particles = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(n_particles)
velocities = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(n_particles)

# 初始化个体最优位置和全局最优位置
pbest_positions = particles.copy()
pbest_values = particles.copy()**2  # 我们初始化为每个粒子当前位置的函数值
gbest_position = pbest_positions[np.argmin(pbest_values)]
gbest_value = min(pbest_values)

# 开始迭代更新
for i in range(n_iterations):
    # 更新速度和位置
    velocities = w*velocities + c1*np.random.rand()*(pbest_positions - particles) + c2*np.random.rand()*(gbest_position - particles)
    particles += velocities
    # 更新个体最优和全局最优
    current_values = particles**2
    pbest_mask = current_values < pbest_values
    pbest_positions[pbest_mask] = particles[pbest_mask]
    pbest_values[pbest_mask] = current_values[pbest_mask]
    if min(current_values) < gbest_value:
        gbest_position = particles[np.argmin(current_values)]
        gbest_value = min(current_values)

print('全局最优位置：', gbest_position)
print('全局最优值：', gbest_value)