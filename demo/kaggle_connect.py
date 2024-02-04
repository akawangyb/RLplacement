# --------------------------------------------------
# 文件名: kaggle_connect
# 创建时间: 2024/2/3 21:42
# 描述: kaggle 学习案例，四个子连接
# 作者: WangYuanbo
# --------------------------------------------------
from kaggle_environments import make, evaluate

# Create the game environment
# Set debug=True to see the errors if your agent refuses to run
env = make("connectx", debug=True)

# List of available default agents
print(list(env.agents))
# Two random agents play one game round
env.run(["random", "random"])

# Show the game
env.render(mode="ipython")