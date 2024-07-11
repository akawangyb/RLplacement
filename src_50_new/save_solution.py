# --------------------------------------------------
# 文件名: save_solution
# 创建时间: 2024/7/11 16:31
# 描述: 持久化模型的专家经验
# 作者: WangYuanbo
# --------------------------------------------------
def baseline_gurobi(env: CustomEnv):
    state, done = env.reset()
    total_reward = torch.zeros(env.container_number)
    while not done:
        action = torch.tensor(env.model_solve())
        state, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward
