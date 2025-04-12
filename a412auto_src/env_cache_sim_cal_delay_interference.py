# --------------------------------------------------
# 文件名: env_joint_place_distribution
# 创建时间: 2025/3/29 18:22
# 描述: 强化学习环境
# 上层强化学习输出容器部署决策y，下层使用RR求解x
# 该环境中，输入是组合动作，输出是当前时隙的时延
# 该环境考虑了干扰的影响
# 作者: WangYuanbo
# --------------------------------------------------
import os
from collections import defaultdict
from typing import List, Dict
from typing import Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


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
        self.current_timestep = 1

        # 动作空间定义
        self.action_space = self._define_action_space()

        self.server_last_placing_states = [-1] * self.servers_number

        self.server_states = {n: {
            'loaded_containers': -1,
            'cpu_used': 0,
            'mem_used': 0,
            'upload_used': 0,
            'download_used': 0,
        } for n in range(self.servers_number)}
        self.current_requests = self.get_current_requests()

        self.state_dim = len(self._build_state())
        self.model = CatBoostRegressor()
        self.model.load_model(r'model_para/catboost_model.bin')

    def _define_action_space(self) -> Dict:
        """定义组合动作空间"""
        return {
            'container_deploy': (len(self.servers), len(self.containers)),  # 每个服务器对每个容器的加载决策
            'request_assign': None  # 动态长度，根据请求数量变化
        }

    def reset(self) -> np.ndarray:
        """重置环境到初始状态"""
        # 重置时隙
        self.current_timestep = 1
        # 生成初始请求批次
        self.current_requests = self.request_generator.generate(timestep=1)
        # 构建初始状态向量
        state = self._build_state()
        return state, False

    def _build_state(self) -> list:
        """构建状态向量
        每个时隙动作网络的输入
        """
        ####################################
        # 当前服务器容量
        # 1 服务器容量 加载速度+内存容量
        server_cap = []
        for server in self.servers:
            for key, values in server.items():
                # if key not in ['server_id', 'tier', 'location', 'storage_capacity']:
                if key in ['b_n', 'mem_capacity']:
                    server_cap.append(server[key])
        #######################################
        # 2 上一时刻的部署结果
        if self.current_timestep == 1:
            self.server_last_placing_states = [-1] * self.servers_number
        else:
            self.server_last_placing_states = [self.server_states[n]['loaded_containers'] \
                                               for n in range(self.servers_number)]
        last_placing = self.server_last_placing_states

        #############################
        #  3 当前的服务部署请求 容器id +保活内存
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

        state = []
        state += server_cap
        state += last_placing
        state += request
        # state += interference
        state += [self.current_timestep * 1.0 / self.max_timesteps]  # 时间
        return state

    def get_state(self):
        """返回当前状态字典"""
        return self._build_state()

    def check(self, action: Tuple):
        """
        检查输入的动作是不是合法的，满足资源约束
        1.每个请求必须分配到唯一位置
        2.每个服务器每个时隙最多部署一个容器
        3.镜像加载依赖
        4.资源容量限制
        :param action:
        """
        container_deploy, request_assign = action
        assert request_assign.shape == (len(self.current_requests), self.servers_number + 1), \
            f'not shape request: {request_assign.shape}, expected {(len(self.current_requests), self.servers_number + 1)}'
        # 约束一
        for r_id, r in enumerate(self.current_requests):
            r_sum = 0
            for s_id in range(self.servers_number + 1):
                r_sum += request_assign[r_id][s_id]
            assert r_sum == 1, f'请求{r_id}分配到多个位置,{request_assign[r_id]}'
        # 约束二
        for s_id in range(self.servers_number):
            c_id = container_deploy[s_id]
            assert 0 <= c_id < self.containers_number, '每个服务器只部署一个容器'
        # 约束三
        for r_id, r in enumerate(self.current_requests):
            # s_id=request_assign[r_id]
            # 相应的边缘服务器上部署了对应的镜像
            for s_id in range(self.servers_number):
                if request_assign[r_id][s_id] == 1:
                    c_id = self.service_type_to_int[r['service_type']]
                    assert c_id == container_deploy[s_id], \
                        '约束3 边缘服务器没有部署相应镜像'
        # 约束4
        for s_id in range(self.servers_number):
            cpu_used = 0
            upload_used = 0
            download_used = 0
            # mem_used = self.h_c_map[container_deploy[s_id]]
            c_id = container_deploy[s_id]
            c_name = self.containers[c_id]
            mem_used = self.h_c_map[c_name]
            for r_id, r in enumerate(self.current_requests):
                if request_assign[r_id][s_id] == 1:
                    cpu_used += r['cpu_demand']
                    upload_used += r['upload_demand']
                    download_used += r['download_demand']
                    mem_used += r['mem_demand']
            assert cpu_used <= self.servers[s_id]['cpu_capacity']
            assert mem_used <= self.servers[s_id]['mem_capacity']
            assert upload_used <= self.servers[s_id]['upload_capacity']
            assert download_used <= self.servers[s_id]['download_capacity']
            # download_demand

    def step(self, action: Tuple) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作并返回新状态、奖励、终止标志和信息
        参数:
            action: 输入是容器的部署动作 + 请求分发动作
            - 'container_deploy': 一维数组（server），表示服务n加载容器c
            - 'request_assign':  二维数组，0，1表示请求r是否由服务器n完成
        """
        self.check(action)
        container_deploy, request_assign = action
        # 阶段1：执行容器部署动作
        load_delay = 0.0
        for server_idx, server in enumerate(self.servers):
            # 更新加载的容器
            new_loaded = container_deploy[server_idx].item()
            # 计算加载变更带来的时延
            prev_loaded = self.server_states[server_idx]['loaded_containers']
            if new_loaded != prev_loaded:
                # 计算镜像加载时延
                service_type = self.containers[new_loaded]
                load_delay += self.h_c_map[service_type] / server['b_n']
            # 更新服务器容器状态
            self.server_states[server_idx]['loaded_containers'] = new_loaded

        # 阶段2：计算资源消耗以及干扰因子
        cpu_used = [0] * self.servers_number
        upload_used = [0] * self.servers_number
        download_used = [0] * self.servers_number
        mem_used = [0] * self.servers_number
        for s_id in range(self.servers_number):
            c_id = container_deploy[s_id]
            c_name = self.containers[c_id]
            mem_used[s_id] = self.h_c_map[c_name]
            for r_id, r in enumerate(self.current_requests):
                if request_assign[r_id][s_id] == 1:
                    cpu_used[s_id] += r['cpu_demand']
                    upload_used[s_id] += r['upload_demand']
                    download_used[s_id] += r['download_demand']
                    mem_used[s_id] += r['mem_demand']
        # 计算干扰因子
        current_alpha = [0.0] * len(self.current_requests)
        for r_id, r in enumerate(self.current_requests):
            for server_idx in range(self.servers_number):
                if request_assign[r_id][server_idx] == 0:
                    continue
                # current_alpha[r_id]=
                demand = [r['cpu_demand'], r['mem_demand'], r['upload_demand'], r['download_demand']]
                supply = [self.servers[server_idx]['cpu_capacity'] - cpu_used[server_idx],
                          self.servers[server_idx]['mem_capacity'] - mem_used[server_idx],
                          self.servers[server_idx]['upload_capacity'] - upload_used[server_idx],
                          self.servers[server_idx]['download_capacity'] - download_used[server_idx]]
                input_vector = [demand[i] / (demand[i]+supply[i]) for i in range(4)] + [0, 0]
                current_alpha[r_id] = self.model.predict(input_vector)

        edge_delay = 0.0
        cloud_delay = 0.0

        for r_id, r in enumerate(self.current_requests):
            # 边缘时延
            for server_idx in range(self.servers_number):
                edge_delay += request_assign[r_id][server_idx] * self.L_e
                edge_delay += request_assign[r_id][server_idx] * \
                              r['compute_delay'] * (1 + current_alpha[r_id])
            # 云时延
            cloud_delay += request_assign[r_id][self.servers_number] * self.L_c
            cloud_delay += request_assign[r_id][self.servers_number] * r['compute_delay']

        # 总奖励
        reward = edge_delay + cloud_delay + load_delay

        # 进入下一时隙
        self.current_timestep += 1
        done = (self.current_timestep > self.max_timesteps)

        # 生成新请求
        self.current_requests = self.request_generator.generate(self.current_timestep)

        # 构建新状态
        next_state = self._build_state()

        # 信息日志
        info = {
            'load_delay': load_delay,
            'edge_delay': edge_delay,
            'cloud_delay': cloud_delay,
            'alpha': sum(current_alpha)/len(current_alpha),
            # 'invalid_load': invalid_load,
        }
        return next_state, reward, done, info

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
