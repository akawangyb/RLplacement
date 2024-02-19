# --------------------------------------------------
# 文件名: ant
# 创建时间: 2024/2/18 22:51
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import numpy as np

# 初始化参数
n_ants = 10
n_cities = 10
evaporation_rate = 0.5
pheromone_constant = 1.0
initial_pheromone = 1.0

distances = np.random.rand(n_cities, n_cities)
pheromones = initial_pheromone * np.ones((n_cities, n_cities))

# 蚁群优化
for i in range(n_ants):
    # 初始化蚂蚁路径和当前城市
    path = []
    current_city = np.random.randint(0, n_cities)
    path.append(current_city)

    # 蚂蚁移动
    for j in range(n_cities - 1):
        # 计算转移概率
        probabilities = pheromones[current_city] / np.sum(pheromones[current_city])

        # 选择下一个城市
        next_city = np.random.choice(range(n_cities), 1, p=probabilities)[0]
        path.append(next_city)
        current_city = next_city

    # 更新信息素
    for k in range(n_cities - 1):
        pheromones[path[k]][path[k + 1]] = (1 - evaporation_rate) * pheromones[path[k]][path[k + 1]] + pheromone_constant / distances[path[k]][path[k + 1]]

# 输出最短路径的长度和路径
min_distance = np.inf
for i in range(n_cities):
    for j in range(i + 1, n_cities):
        if distances[i][j] < min_distance:
            min_distance = distances[i][j]
            min_distance_path = (i, j)
print(f"最短路径长度为: {min_distance}")
print(f"最短路径为: {min_distance_path}")