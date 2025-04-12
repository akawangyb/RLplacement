# --------------------------------------------------
# 文件名: env_joint_place_distribution
# 创建时间: 2025/3/29 18:22
# 描述: 多智能体容器部署强化学习环境
# 作者: WangYuanbo
# --------------------------------------------------
import os
from collections import defaultdict
from typing import List, Dict
from typing import Tuple
import numpy as np
import pandas as pd
import torch


# 定义好强化学习的3要素
# 状态空间
# - 服务器状态：每个服务器的CPU、内存、带宽容量，已加载的容器集合
# - 请求状态： 当前时隙的服务请求集合（请求数量动态变化），每个请求的容器类型、资源需求（CPU、内存、带宽
# - 历史动作：上一时隙的容器加载何请求分配结果
# 动作空间
# 容器部署决策，为每一个服务器选择加载哪些容器
# 请求分配决策， 为每一个请求分配目标（边缘服务器和云）
# 奖励函数
# 负的总时延


class RequestGenerator:
    def __init__(self, df_request):
        """
        参数:
            service_probs: 服务类型概率分布，如 {'ai_inference': 0.6, 'video_transcode': 0.4}
            lambda_poisson: 泊松分布参数（每个时隙平均请求数）
            time_slots: 总时隙数
        """
        requests = df_request.to_dict('records')
        self.df_request = requests
        # 按time_slot分组
        self.timeslot_dict = defaultdict(list)
        for req in self.df_request:
            timeslot = int(req["time_slot"])
            self.timeslot_dict[timeslot].append(req)

    def generate(self, timestep: int) -> List[Dict]:
        # 过滤当前时隙的请求（假设time_slot列存在）
        current_requests_df = self.timeslot_dict[timestep]

        # 转换为字典列表（关键步骤）
        requests = current_requests_df

        # 类型检查与保证
        assert isinstance(requests, list), "输出必须是列表"
        for req in requests:
            assert isinstance(req, dict), "列表元素必须是字典"

        return requests


class EdgeDeploymentEnv:
    def __init__(self, dataset_dir):
        """
        参数:
            servers: 边缘服务器列表，每个元素包含资源容量和镜像加载速度
            containers: 容器类型列表
            max_timesteps: 最大时隙数
            request_generator: 请求生成器（动态生成每个时隙的请求）
        """
        self.penalty = 50
        self.steps = 0
        self.L_e = 50
        self.L_c = 300
        self.dataset_dir = dataset_dir
        # 从数据集文件中获取
        container_requests_path = os.path.join(dataset_dir, 'container_requests.csv')
        edge_servers_path = os.path.join(dataset_dir, 'edge_servers.csv')

        # 读取请求数据（已包含时隙信息）
        df_requests = pd.read_csv(container_requests_path)

        # 读取服务器数据
        df_servers = pd.read_csv(edge_servers_path)
        servers = df_servers.to_dict('records')

        # 定义常量参数
        time_slots = sorted(df_requests['time_slot'].unique())  # 1~24时隙

        self.servers = servers
        self.containers = list(df_requests['service_type'].unique())

        self.servers_number = len(servers)
        self.containers_number = len(self.containers)

        # 按 service_type 分组并提取唯一 h_c
        # self.h_c_map = df.groupby("service_type")["h_c"].unique().reset_index()

        # 提取唯一的 service_type 和 h_c 组合
        self.h_c_map = df_requests[["service_type", "h_c"]].drop_duplicates().set_index("service_type")[
            "h_c"].to_dict()

        # 环境参数
        self.time_slots = time_slots
        self.max_timesteps = len(time_slots)

        self.request_generator = RequestGenerator(df_requests)

        # 单个时隙的最大请求数量
        # 找到请求数量最大的时隙
        # 按time_slot分组，统计每个时隙的请求数量
        requests_per_slot = df_requests.groupby("time_slot").size().reset_index(name="request_count")
        max_requests = requests_per_slot["request_count"].max()
        self.max_requests = max_requests

        self.service_type_to_int = {st: idx for idx, st in enumerate(self.containers)}

        # 状态空间
        # N*4+C*N + R_max*7(4+类型+保活内存+延迟) + C*N(干扰情况)+时隙
        self.current_timestep = 0

        # 动作空间定义
        self.action_space = self._define_action_space()

        self.server_last_placing_states = [-1] * self.servers_number

        self.server_states = {server_id: {
            'loaded_containers': -1,
            'cpu_used': 0,
            'mem_used': 0,
            'upload_used': 0,
            'download_used': 0,
        } for server_id in range(self.servers_number)}
        obs, state, _ = self.reset()
        assert len(obs) == self.servers_number, \
            f'Observations must have the same number, obs len {len(obs)},'
        self.obs_dim = len(obs[0])

        self.state_dim = len(state)

    def _define_action_space(self) -> Dict:
        """定义组合动作空间"""
        return {
            'container_deploy': (len(self.servers), len(self.containers)),  # 每个服务器对每个容器的加载决策
            'request_assign': None  # 动态长度，根据请求数量变化
        }

    def reset(self) -> Tuple:
        """重置环境到初始状态"""
        # 重置时隙
        self.current_timestep = 0
        # 生成初始请求批次
        self.current_requests = self.request_generator.generate(timestep=0)
        # 构建初始状态向量
        obs, state = self._build_state()
        return obs, state, False

    def _build_state(self) -> Tuple[list, list]:
        """构建状态向量
        每个时隙动作网络的输入
        第一个list表示每一个智能体的观察空间
        第二list表示全局观察空间
        """
        #######################################
        # 2 上一时刻的部署结果
        if self.current_timestep == 0:
            self.server_last_placing_states = [-1] * self.servers_number
        else :
            self.server_last_placing_states = [self.server_states[n]['loaded_containers'] \
                                             for n in range(self.servers_number)]
        last_placing = self.server_last_placing_states
        ####################################
        # 当前服务器容量
        # 1 服务器容量
        server_cap = []
        obs = []
        for server_id, server in enumerate(self.servers):
            obs_i = [last_placing[server_id]]
            for key, values in server.items():
                if key in ['b_n', 'mem_capacity']:
                    server_cap.append(server[key])
                    obs_i.append(server[key])
            obs_i.append(self.current_timestep * 1.0 / self.max_timesteps)
            obs.append(obs_i)

        #############################
        #  3 当前的服务部署请求
        temp_requests = self.current_requests
        request = []
        for rs in temp_requests:
            for key, values in rs.items():
                # if key in ['request_id', 'image_size', 'time_slot']:
                if key not in ['service_type', 'h_c']:
                    continue
                if key == 'service_type':
                    request.append(self.service_type_to_int[values])
                else:
                    request.append(values)
        # 把请求补全,每个请求
        # 每个请求的数据是容器id+保活内存大小, 请求的总长度是133
        number = self.max_requests - len(temp_requests)
        for i in range(number):
            request += [-1] * 2
        for id in range(self.servers_number):
            obs[id].extend(request)

        # 4 上一个时隙的干扰状态
        # self.last_interference = [[0] * self.servers_number] * self.containers_number
        # 把他拉伸
        # temp = []
        # for ele in self.last_interference:
        #     temp += ele
        # interference = temp
        #######################################
        state = []
        for obs_i in obs:
            state.extend(obs_i)
        return obs, state

    def get_state(self):
        """返回当前状态字典"""
        return self._build_state()

    def compute_constraint_violation(self, action: np.ndarray):
        """
        根据输入的动作计算资源违反量
        :param action:
        :return:
        """
        container_deploy = action
        # 阶段1：执行容器部署动作
        new_mem_used = [0] * self.servers_number
        for n in range(self.servers_number):
            d_m = 0
            for c in range(self.containers_number):
                d_m += container_deploy[n][c] * \
                       self.h_c_map[self.containers[c]]
            new_mem_used[n] = d_m
        violation = [0] * self.servers_number
        for server_id, ele in enumerate(new_mem_used):
            max_mem = self.servers[server_id]['mem_capacity']
            violation[server_id] = max(0, ele * 1.0 / max_mem - 1)
            # violation[server_id] = ele * 1.0 / max_mem
        # return violation
        return violation

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作并返回新状态、奖励、终止标志和信息
        参数:
            action: 包含两个键值:
            - 'container_deploy': 二维数组（server×container），0/1表示是否加载
        """
        self.steps += 1
        container_deploy = action
        # 阶段1：执行容器部署动作
        container_reward = 0.0
        for server_idx, server in enumerate(self.servers):
            server_name = server['server_id']
            # 更新加载的容器
            new_loaded = container_deploy[server_idx]
            # 计算加载变更带来的时延 #越少越好
            prev_loaded = self.server_states[server_idx]['loaded_containers']
            # load_changes = np.array((new_loaded == 1) & (prev_loaded == 0))
            if new_loaded == prev_loaded:
                # 计算镜像加载时延
                # container_reward -= self.h_c_map[self.containers[c_idx]] / server['b_n']
                container_reward += 1
            # 更新服务器容器状态
            self.server_states[server_idx]['loaded_containers'] = new_loaded.item()

        # 更新服务器内存
        # invalid_load = 0
        invalid_load = [0] * self.servers_number
        for n in range(self.servers_number):
            d_m = 0
            server_name = self.servers[n]['server_id']
            c = int(container_deploy[n])
            if self.server_states[n]['loaded_containers'] != c:
                if d_m + self.h_c_map[self.containers[c]] > \
                        self.servers[n]['mem_capacity']:
                    invalid_load[n] += 1
                else:
                    d_m += self.h_c_map[self.containers[c]]
            if d_m > self.servers[n]['mem_capacity']:
                self.server_states[n]['mem_used'] = 0
            else:
                self.server_states[n]['mem_used'] = d_m

        # 计算缓存命中数量,越多越好
        hits = [0] * self.servers_number
        # 分别计算每个服务器的奖励
        rewards = [0] * self.servers_number
        for rs in self.current_requests:
            container_id = self.service_type_to_int[rs['service_type']]
            for server_idx in range(self.servers_number):
                if container_deploy[server_idx] == container_id:
                    hits[server_idx] += 1

        # 总奖励
        for n in range(self.servers_number):
            rewards[n] = hits[n] - self.penalty * invalid_load[n]
        # 进入下一时隙
        self.current_timestep += 1
        done = (self.current_timestep >= self.max_timesteps)

        # 生成新请求
        self.current_requests = self.request_generator.generate(self.current_timestep)

        # 构建新状态
        obs, next_state = self._build_state()
        # 信息日志
        info = {
            'container_reward': container_reward,
            'invalid_load': invalid_load,
        }
        return obs, next_state, rewards, done, info

    def get_current_requests(self):
        """获取当前时隙的请求列表"""
        self.current_requests = self.request_generator.generate(self.current_timestep)
        return self.current_requests


if __name__ == '__main__':
    # 初始化环境
    # 测试一下
    env = EdgeDeploymentEnv(dataset_dir='data/')
    # 训练循环
    state = env.reset()
    done = False
    total_delay = 0
    while not done:
        # 假设智能体生成动作
        action = {
            'container_deploy': np.random.randint(0, 2, (3, 4)),
            'request_assign': ['cloud'] * 100
        }
        next_state, reward, done, info = env.step(action)
        total_delay += reward
        print(reward)
    print(total_delay)
