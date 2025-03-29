# --------------------------------------------------
# 文件名: env_joint_place_distribution
# 创建时间: 2025/3/29 18:22
# 描述: 强化学习环境
# 作者: WangYuanbo
# --------------------------------------------------
import os
from collections import defaultdict
from typing import List, Dict
from typing import Tuple

import numpy as np
import pandas as pd


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
        """生成指定时隙的请求列表"""
        return self.df_request[timestep]


class EdgeDeploymentEnv:
    def __init__(self, dataset_dir):
        """
        参数:
            servers: 边缘服务器列表，每个元素包含资源容量和镜像加载速度
            containers: 容器类型列表
            max_timesteps: 最大时隙数
            request_generator: 请求生成器（动态生成每个时隙的请求）
        """
        self.penalty = 20000
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

        # 动作空间定义
        self.action_space = self._define_action_space()

        # 状态空间维度
        self.state_dim = self._calculate_state_dim()

    def _define_action_space(self) -> Dict:
        """定义组合动作空间"""
        return {
            'container_deploy': (len(self.servers), len(self.containers)),  # 每个服务器对每个容器的加载决策
            'request_assign': None  # 动态长度，根据请求数量变化
        }

    def _calculate_state_dim(self) -> int:
        """计算状态向量维度"""
        # 服务器状态维度: [cpu_usage, mem_usage, upload, download] + 容器加载状态
        server_dim = 4 + len(self.containers)
        # 请求状态维度: [container_type, cpu_demand, mem_demand, upload, download]
        request_dim = 5
        # 全局状态: 当前时隙
        return len(self.servers) * server_dim + self.max_requests * request_dim + 1

    def reset(self) -> np.ndarray:
        """重置环境到初始状态"""
        # 重置时隙
        self.current_timestep = 0
        # 初始化服务器资源状态
        self.server_states = {
            s['server_id']: {
                'cpu_used': 0.0,
                'mem_used': 0.0,
                'upload_used': 0.0,
                'download_used': 0.0,
                'loaded_containers': np.zeros(len(self.containers), dtype=int)
            } for s in self.servers
        }
        # 生成初始请求批次
        self.current_requests = self.request_generator.generate(timestep=0)

        # 构建初始状态向量
        state = self._build_state()
        return state

    def _build_state(self) -> np.ndarray:
        """构建状态向量"""
        state = []

        # 服务器状态编码
        for s in self.servers:
            s_id = s['server_id']
            state += [
                self.server_states[s_id]['cpu_used'] / s['cpu_capacity'],
                self.server_states[s_id]['mem_used'] / s['mem_capacity'],
                self.server_states[s_id]['upload_used'] / s['upload_capacity'],
                self.server_states[s_id]['download_used'] / s['download_capacity'],
                *self.server_states[s_id]['loaded_containers']  # 容器加载状态
            ]

        # 请求状态编码（填充至最大长度）
        max_requests = self.max_requests  # 假设最大请求数
        self.current_requests = [self.current_requests]
        for req in self.current_requests:
            state += [
                self.service_type_to_int[req['service_type']],  # 需转换为整数索引
                req['cpu_demand'],
                req['mem_demand'],
                req['upload_demand'],
                req['download_demand']
            ]
        # 填充不足部分
        state += [0] * (5 * (max_requests - len(self.current_requests)))

        # 添加时隙信息
        state.append(self.current_timestep / self.max_timesteps)

        return np.array(state, dtype=np.float32)

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作并返回新状态、奖励、终止标志和信息

        参数:
            action: 包含两个键值:
                - 'container_deploy': 二维数组（server×container），0/1表示是否加载
                - 'request_assign': 列表，每个元素为请求的目标服务器ID或'cloud'
        """
        # 阶段1：执行容器部署动作
        container_reward = 0.0
        for server_idx, server in enumerate(self.servers):
            s_id = server['server_id']
            # 更新加载的容器
            new_loaded = action['container_deploy'][server_idx]
            # 计算加载变更带来的时延
            prev_loaded = self.server_states[s_id]['loaded_containers']
            load_changes = np.where(new_loaded != prev_loaded)[0]
            for c_idx in load_changes:
                if new_loaded[c_idx] == 1:
                    # 计算镜像加载时延
                    container_reward -= self.h_c_map[self.containers[c_idx]] / server['b_n']
            # 更新服务器容器状态
            self.server_states[s_id]['loaded_containers'] = new_loaded

        # 阶段2：执行请求分配动作
        compute_reward = 0.0
        prop_reward = 0.0
        invalid_actions = 0

        for req_idx, req in enumerate(self.current_requests):
            target = action['request_assign'][req_idx]
            # 检查目标合法性
            if target != 'cloud':
                s_id = target
                # 检查容器是否已加载
                c_idx = self.containers.index(req['container_type'])
                if not self.server_states[s_id]['loaded_containers'][c_idx]:
                    invalid_actions += 1
                    target = 'cloud'  # 强制分配到云

                # 检查资源是否足够
                server = next(s for s in self.servers if s['server_id'] == s_id)
                if (self.server_states[s_id]['cpu_used'] + req['cpu_demand'] > server['cpu_capacity'] or
                        self.server_states[s_id]['mem_used'] + req['mem_demand'] > server['mem_capacity']):
                    invalid_actions += 1
                    target = 'cloud'

            # 计算时延
            if target == 'cloud':
                compute_reward -= req['compute_delay']
                prop_reward -= self.L_c
            else:
                # 边缘计算时延（含干扰）
                compute_reward -= req['compute_delay'] * (1 + self._get_interference_coeff(target))
                prop_reward -= self.L_e

            # 更新服务器资源使用（仅边缘目标）
            if target != 'cloud':
                self.server_states[target]['cpu_used'] += req['cpu_demand']
                self.server_states[target]['mem_used'] += req['mem_demand']

        # 总奖励
        reward = container_reward + compute_reward + prop_reward - self.penalty * invalid_actions

        # 进入下一时隙
        self.current_timestep += 1
        done = (self.current_timestep >= self.max_timesteps)

        # 生成新请求
        self.current_requests = self.request_generator.generate(self.current_timestep)

        # 构建新状态
        next_state = self._build_state()

        # 信息日志
        info = {
            'container_reward': container_reward,
            'compute_reward': compute_reward,
            'prop_reward': prop_reward,
            'invalid_actions': invalid_actions
        }
        return next_state, reward, done, info

    def _get_interference_coeff(self, server_id: str) -> float:
        """根据服务器负载计算干扰系数"""
        server = next(s for s in self.servers if s['server_id'] == server_id)
        cpu_util = self.server_states[server_id]['cpu_used'] / server['cpu_capacity']
        return 0.5 * cpu_util  # 示例公式


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
