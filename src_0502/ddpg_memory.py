# --------------------------------------------------
# 文件名: ddpg_memory
# 创建时间: 2024/5/15 14:58
# 描述: ddpg算法记忆池
# 作者: WangYuanbo
# --------------------------------------------------

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_size, actions, buffer_size, device='cpu'):
        # state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float, device=device)
        self.action = [torch.empty((buffer_size, *action_size), dtype=torch.float, device=device) for action_size in
                       actions]
        self.reward = torch.empty(buffer_size, 1, dtype=torch.float, device=device)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float, device=device)
        self.done = torch.empty(buffer_size,1, dtype=torch.int, device=device)
        self.device = device
        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        state, action, reward, next_state, done = transition
        # store transition in the buffer
        self.state[self.count] = state
        # 假设actions传过来是一个字典
        for i in range(len(self.action)):
            self.action[i][self.count] = action[i]

        self.reward[self.count] = reward
        self.next_state[self.count] = next_state
        self.done[self.count] = done

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size
        sample_idxs = np.random.choice(self.real_size, batch_size, replace=False)
        batch = (
            self.state[sample_idxs],
            [chosen_action[sample_idxs] for chosen_action in self.action],
            self.reward[sample_idxs],
            self.next_state[sample_idxs],
            self.done[sample_idxs]
        )
        return batch
