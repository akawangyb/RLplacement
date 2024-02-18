import gym
from gym import spaces, core
import numpy as np

# 实验参数设置
# 对于服务器 ，假设都是同构的，cpu为128，mem为64，共 5个
# 对于一个容器，随机生成一些数据 cpu为[1,20],mem为[1,4] 共有200个
node_max_cpu = 128
node_max_mem = 64
node_number = 5


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # 定义状态空间
        self.low_state = np.array([-1.0, -1.0])
        self.high_state = np.array([1.0, 1.0])
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        # 定义动作空间
        self.low_action = np.array([-0.1, -0.1])
        self.high_action = np.array([0.1, 0.1])
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)

        # 初始化环境参数
        self.state = np.zeros(2)  # 初始状态

        # # 定义动作空间
        # self.action_space = spaces.Discrete(6)  # 11个离散动作空间，0-9表示放入前十个
        # # 定义状态空间
        # node_state = [node_number, node_max_cpu, node_max_mem]
        # container_state = [128, 100, 50]
        # # 容器状态的最后一维表示容器部署的收益
        # # 状态空间是由（节点状态，容器状态）构成的
        # self.observation_space = spaces.Tuple([
        #     spaces.MultiDiscrete(node_state),
        #     spaces.MultiDiscrete(container_state)])
        #
        # self.state = 0

    def step(self, action):
        # # action=11 表示不放置这个容器，直接返回
        # if action == 11:
        #     return self.state, 0, False, {}
        #
        # # 执行动作并计算奖励
        # reward = self._calculate_reward(self.state, action)
        #
        # # 更新环境状态
        # self.state =
        #
        # # 返回新的状态，奖励，是否结束，以及其他信息
        # # 这里的其他信息主要是为了偏于调试
        # return self.state, reward, False, {}
        # 执行动作并返回新状态、奖励、是否终止和额外信息
        self.state = np.clip(self.state + action, self.low_state, self.high_state)
        # np.clip(arr,min_value,max_value) 把arr中的数值限制在[min_value,max_value]之间
        reward = -np.sum(np.abs(self.state))  # 举例：负奖励，目标是让状态接近零
        done = False  # 可根据具体条件设置是否终止
        info = {}  # 可以存放额外的信息
        return self.state, reward, done, info

    def reset(self):
        # 重置环境到初始状态
        self.state = np.array([np.random.randint(0, 100)])
        return self.state

    def _calculate_reward(self, state, action):
        #
        reward = state[1][3]
        # 这里需要检查一下状态空间直接访问是否正确的
        return reward


if __name__ == '__main__':
    env = CustomEnv()
    observation = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        print("Action:", action, "State:", state, "Reward:", reward, "Done:", done)
    env.close()
