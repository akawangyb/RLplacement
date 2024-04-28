import random

import torch
from memory.tree import SumTree


# from memory.utils import device


class PrioritizedReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, device='cpu', eps=1e-2, alpha=0.1, beta=0.1):
        self.tree = SumTree(size=buffer_size)
        self.device = device
        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # 为什么要创建空张量？
        # 创建一块连续的内存空间，用于存储将被频繁访问和修改的数据，可以提高处理速度。
        # 预先分配所需的空间，这样在后续的操作中就不必再进行昂贵的内存重新分配操作。
        # 这里state,就是buffer_size*state_size的张量，
        # transition: state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size


def add(self, transition):
    state, action, reward, next_state, done = transition
    # 为什么新添加的记忆要使用最大权重？
    # 增加新记忆的新奇性，提高训练效率
    # store transition index with maximum priority in sum tree
    self.tree.add(self.max_priority, self.count)

    # store transition in the buffer
    # 把状态转移以tensor的形式存起来
    self.state[self.count] = torch.as_tensor(state)
    self.action[self.count] = torch.as_tensor(action)
    self.reward[self.count] = torch.as_tensor(reward)
    self.next_state[self.count] = torch.as_tensor(next_state)
    self.done[self.count] = torch.as_tensor(done)

    # update counters
    self.count = (self.count + 1) % self.size
    self.real_size = min(self.size, self.real_size + 1)


def sample(self, batch_size):
    assert self.real_size >= batch_size, "buffer contains less samples than batch size"

    sample_idxs, tree_idxs = [], []
    priorities = torch.empty(batch_size, 1, dtype=torch.float)

    # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
    # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
    # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
    segment = self.tree.total / batch_size
    for i in range(batch_size):
        a, b = segment * i, segment * (i + 1)

        cumsum = random.uniform(a, b)
        # sample_idx is a sample index in buffer, needed further to sample actual transitions
        # tree_idx is a index of a sample in the tree, needed further to update priorities
        tree_idx, priority, sample_idx = self.tree.get(cumsum)

        priorities[i] = priority
        tree_idxs.append(tree_idx)
        sample_idxs.append(sample_idx)

    # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
    # where p_i > 0 is the priority of transition i. (Section 3.3)
    probs = priorities / self.tree.total

    # The estimation of the expected value with stochastic updates relies on those updates corresponding
    # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
    # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
    # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
    # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
    # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
    # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
    # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
    # update downwards (Section 3.4, first paragraph)
    weights = (self.real_size * probs) ** -self.beta

    # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
    # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
    # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
    weights = weights / weights.max()

    batch = (
        self.state[sample_idxs].to(self.device),
        self.action[sample_idxs].to(self.device),
        self.reward[sample_idxs].to(self.device),
        self.next_state[sample_idxs].to(self.device),
        self.done[sample_idxs].to(self.device)
    )
    return batch, weights, tree_idxs


def update_priorities(self, data_idxs, priorities):
    if isinstance(priorities, torch.Tensor):
        priorities = priorities.detach().cpu().numpy()

    for data_idx, priority in zip(data_idxs, priorities):
        # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
        # where eps is a small positive constant that prevents the edge-case of transitions not being
        # revisited once their error is zero. (Section 3.3)
        priority = (priority + self.eps) ** self.alpha

        self.tree.update(data_idx, priority)
        self.max_priority = max(self.max_priority, priority)


# 多智能体优先记忆池
class MAPrioritizedReplayBuffer:
    def __init__(self, state_size, placing_action_size, routing_action_size, buffer_size, placing_agent_number,
                 routing_agent_number, eps=1e-2, device='cpu',
                 alpha=0.1, beta=0.1):
        self.tree = SumTree(size=buffer_size)
        self.device = device
        agent_number = placing_agent_number + routing_agent_number

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # 为什么要创建空张量？
        # 创建一块连续的内存空间，用于存储将被频繁访问和修改的数据，可以提高处理速度。
        # 预先分配所需的空间，这样在后续的操作中就不必再进行昂贵的内存重新分配操作。
        # 这里state,就是buffer_size*state_size的张量，
        # transition: state, action, reward, next_state, done
        # 所有的智能体公用state,next_state,done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.placing_action = torch.empty(buffer_size, placing_agent_number, placing_action_size, dtype=torch.float)
        self.routing_action = torch.empty(buffer_size, routing_agent_number, routing_action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, agent_number, dtype=torch.float)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        # action,reward是一个list
        state, placing_action, routing_action, reward, next_state, done = transition

        # 为什么新添加的记忆要使用最大权重？
        # 增加新记忆的新奇性，提高训练效率
        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        # 把状态转移以tensor的形式存起来
        self.state[self.count] = torch.as_tensor(state)
        # self.placing_action[self.count] = torch.as_tensor(placing_action)
        # self.routing_action[self.count] = torch.as_tensor(routing_action)
        self.placing_action[self.count] = torch.vstack(placing_action)
        self.routing_action[self.count] = torch.vstack(routing_action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs].to(self.device),
            self.placing_action[sample_idxs].to(self.device),
            self.routing_action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device)
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()
        # print(type(priorities), priorities)
        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)


class MAPrioritizedTensorReplayBuffer:
    def __init__(self, state_size, placing_action_size, routing_action_size, buffer_size, placing_agent_number,
                 routing_agent_number, eps=1e-2, device='cpu',
                 alpha=0.1, beta=0.1):
        self.tree = SumTree(size=buffer_size)
        self.device = device
        agent_number = placing_agent_number + routing_agent_number

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # 为什么要创建空张量？
        # 创建一块连续的内存空间，用于存储将被频繁访问和修改的数据，可以提高处理速度。
        # 预先分配所需的空间，这样在后续的操作中就不必再进行昂贵的内存重新分配操作。
        # 这里state,就是buffer_size*state_size的张量，
        # transition: state, action, reward, next_state, done
        # 所有的智能体公用state,next_state,done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.placing_action = torch.empty(buffer_size, placing_agent_number, placing_action_size, dtype=torch.float)
        self.routing_action = torch.empty(buffer_size, routing_agent_number, routing_action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, agent_number, dtype=torch.float)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        # action,reward是一个list
        state, placing_action, routing_action, reward, next_state, done = transition

        # 为什么新添加的记忆要使用最大权重？
        # 增加新记忆的新奇性，提高训练效率
        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        # 把状态转移以tensor的形式存起来
        self.state[self.count] = state

        self.placing_action[self.count] = torch.vstack(placing_action)
        self.routing_action[self.count] = torch.vstack(routing_action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = next_state
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs].to(self.device),
            self.placing_action[sample_idxs].to(self.device),
            self.routing_action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device)
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()
        # print(type(priorities), priorities)
        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
