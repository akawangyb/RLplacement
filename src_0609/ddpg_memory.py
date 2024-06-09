# --------------------------------------------------
# 文件名: ddpg_memory
# 创建时间: 2024/6/5 16:00
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, device='cpu'):
        # state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float, device=device)
        self.action = torch.empty((buffer_size, *action_size), dtype=torch.float, device=device)
        self.reward = torch.empty(buffer_size, 1, dtype=torch.float, device=device)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float, device=device)
        self.done = torch.empty(buffer_size, 1, dtype=torch.int, device=device)
        self.device = device
        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        state, action, reward, next_state, done = transition
        # store transition in the buffer
        self.state[self.count] = state
        self.action[self.count] = action

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
            self.action[sample_idxs],
            self.reward[sample_idxs],
            self.next_state[sample_idxs],
            self.done[sample_idxs]
        )
        return batch


class MAReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, agent_number, device='cpu'):
        # state, action, reward, next_state, done
        self.agent_number = agent_number
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float, device=device)
        self.action = [torch.empty(buffer_size, action_size, dtype=torch.float, device=device)
                       for _ in range(self.agent_number)]
        self.reward = [torch.empty(buffer_size, 1, dtype=torch.float, device=device)
                       for _ in range(self.agent_number)]
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float, device=device)
        self.done = torch.empty(buffer_size, 1, dtype=torch.int, device=device)
        self.device = device
        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        state, actions, rewards, next_state, done = transition
        # store transition in the buffer
        self.state[self.count] = state
        for id, action in enumerate(actions):
            self.action[id][self.count] = action

        for id, reward in enumerate(rewards):
            self.reward[id][self.count] = reward
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
            [self.action[id][sample_idxs] for id in range(self.agent_number)],
            [self.reward[id][sample_idxs] for id in range(self.agent_number)],
            self.next_state[sample_idxs],
            self.done[sample_idxs]
        )
        return batch
