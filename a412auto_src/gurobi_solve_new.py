# --------------------------------------------------
# 文件名: gurobi_solve
# 创建时间: 2025/3/29 10:35
# 描述: 用gurobi求解全局优化问题
# 作者: WangYuanbo
# --------------------------------------------------
import csv
import os
import pickle
from collections import namedtuple

import gurobipy as gp
import numpy as np
import pandas as pd
import yaml
from gurobipy import GRB

with open('train_config.yaml', 'r', encoding='utf-8') as f:
    config_data = yaml.safe_load(f)

Config = namedtuple('Config',
                    [
                        'GAMMA',
                        'LAMBDA',
                        'CLIP_EPS',
                        'LR_ACTOR',
                        'LR_CRITIC',
                        'BATCH_SIZE',
                        'EPOCHS',
                        'MAX_GRAD_NORM',
                        'C_MAX',
                        'data_dir',
                        'multiplier_lr',
                        'lag_multiplier',
                    ])
config = Config(**config_data)
edge_prop_delay = 50
cloud_prop_delay = 300
# =============================
# 1. 数据准备
# =============================

dataset_path = config.data_dir
# 读取请求数据（已包含时隙信息）
df_requests = os.path.join(dataset_path, 'container_requests.csv')
df_requests = pd.read_csv(df_requests)
requests = df_requests.to_dict('records')
# print(df_requests)

# 读取服务器数据
df_servers = os.path.join(dataset_path, 'edge_servers.csv')
df_servers = pd.read_csv(df_servers)
servers = df_servers.to_dict('records')

# 定义常量参数
time_slots = sorted(df_requests['time_slot'].unique())  # 1~24时隙
service_types = df_requests['service_type'].unique()
# 提取唯一的 service_type 和 h_c 组合
service_hc_map = df_requests[["service_type", "h_c"]].drop_duplicates().set_index("service_type")["h_c"].to_dict()
# print(service_hc_map)

server_index_map = {}
with open(os.path.join(dataset_path, 'edge_servers.csv')) as f:
    reader = csv.reader(f)
    next(reader)  # 跳过标题行
    for index, row in enumerate(reader):
        server_id = row[0]
        server_index_map[server_id] = index
server_index_map['cloud']=len(servers)

server_num = len(servers)
container_num = len(service_types)

# =============================
# 2. 创建Gurobi模型
# =============================
model = gp.Model("EdgeContainerDeployment")

# =============================
# 3. 定义决策变量
# =============================
# 请求分配变量 x[r][k][t]
x = {}
for r in requests:
    req_id = r['request_id']
    t = r['time_slot']
    x[req_id] = model.addVars(
        [s['server_id'] for s in servers] + ['cloud'],
        vtype=GRB.BINARY,
        name=f"x_{req_id}"
    )

# 镜像加载变量 y[c][n][t]
containers = list(df_requests['service_type'].unique())
y = model.addVars(
    containers,
    [s['server_id'] for s in servers],
    time_slots,
    vtype=GRB.BINARY,
    name="y"
)

# 镜像加载触发变量 z[c][n][t] (0-1变化时触发)
z = model.addVars(
    containers,
    [s['server_id'] for s in servers],
    time_slots,
    vtype=GRB.BINARY,
    name="z"
)

# =============================
# 4. 定义目标函数
# =============================
total_cost = gp.QuadExpr()

# 镜像加载时延成本
# 定义目标函数中的镜像加载时延项
for s in servers:
    server_id = s['server_id']
    b_n = s['b_n']  # 加载速度（MB/s）
    for c in containers:
        h_c = service_hc_map[c]  # 容器保活内存（MB）
        for t in time_slots:
            # 加载时延 = h_c / b_n * z[c,n,t]
            total_cost += (h_c / b_n) * z[c, server_id, t]

# 计算时延和传播时延
for r in requests:
    req_id = r['request_id']
    c = r['service_type']
    t = r['time_slot']

    # 边缘计算时延
    for s in servers:
        server_id = s['server_id']
        # 不考虑干扰问题
        compute_delay = r["compute_delay"]
        total_cost += x[req_id][server_id] * (compute_delay + edge_prop_delay)
    # 云端计算惩罚
    compute_delay = r['compute_delay']
    # total_cost += x[req_id]['cloud'] * (compute_delay_cloud + cloud_prop_delay)
    total_cost += x[req_id]['cloud'] * (compute_delay + cloud_prop_delay)

model.setObjective(total_cost, GRB.MINIMIZE)
# =============================
# 5. 定义约束条件
# =============================
# 约束1：每个请求必须分配到唯一位置
for r in requests:
    req_id = r['request_id']
    model.addConstr(
        gp.quicksum(x[req_id][k] for k in x[req_id]) == 1,
        name=f"uniq_alloc_{req_id}"
    )
    # print(x[req_id].keys())  # 应输出服务器ID列表 + 'cloud'

# 约束2：资源容量限制
for s in servers:
    server_id = s['server_id']
    for t in time_slots:
        # 获取当前时隙的请求
        slot_requests = [r for r in requests if r['time_slot'] == t]
        # CPU约束
        cpu_usage = gp.quicksum(
            r['cpu_demand'] * x[r['request_id']][server_id]
            for r in slot_requests
        )
        model.addConstr(cpu_usage <= s['cpu_capacity'], name=f"cpu_{server_id}_{t}")

        # 内存约束（动态内存+镜像内存）
        mem_usage = gp.quicksum(
            r['mem_demand'] * x[r['request_id']][server_id]
            for r in slot_requests
        )
        mem_usage += gp.quicksum(
            # y[c, server_id, t] * df_requests[df_requests['service_type'] == c]['h_c'].mean()
            y[c, server_id, t] * service_hc_map[c]
            for c in containers
        )
        model.addConstr(mem_usage <= s['mem_capacity'], name=f"mem_{server_id}_{t}")

        # 服务器上行带宽约束
        model.addConstr(
            gp.quicksum(
                r['upload_demand'] * x[r['request_id']][server_id]
                for r in slot_requests
            ) <= s['upload_capacity'],
            name=f"upload_{server_id}_{t}"
        )

        # 服务器下行带宽约束
        model.addConstr(
            gp.quicksum(
                r['download_demand'] * x[r['request_id']][server_id]
                for r in slot_requests
            ) <= s['download_capacity'],
            name=f"download_{server_id}_{t}"
        )

# 约束3：镜像加载依赖
for r in requests:
    req_id = r['request_id']
    c = r['service_type']
    t = r['time_slot']
    for s in servers:
        server_id = s['server_id']
        model.addConstr(
            x[req_id][server_id] <= y[c, server_id, t],
            name=f"img_dep_{req_id}_{server_id}_{t}"
        )

# 约束4：镜像状态变化约束
for s in servers:
    server_id = s['server_id']
    for c in containers:
        for t in time_slots:
            if t == 1:
                model.addConstr(z[c, server_id, t] == y[c, server_id, t])
            else:
                # z[c,n,t] = y[c,n,t] AND NOT y[c,n,t-1]
                model.addConstr(z[c, server_id, t] <= y[c, server_id, t])
                model.addConstr(z[c, server_id, t] <= 1 - y[c, server_id, t - 1])
                model.addConstr(z[c, server_id, t] >= y[c, server_id, t] - y[c, server_id, t - 1])
# =============================
# 5. 定义约束条件（新增单容器部署约束）
# =============================

# 约束5：每个服务器每个时隙最多部署一个容器
for t in time_slots:
    for s in servers:
        server_id = s['server_id']
        # 所有容器在该服务器该时隙的 y 变量之和 ≤ 1
        model.addConstr(
            gp.quicksum(y[c, server_id, t] for c in containers) == 1,
            name=f"single_container_{server_id}_{t}"
        )
# =============================
# 6. 求解与结果分析
# =============================
# model.Params.TimeLimit = 60  # 10分钟限制
model.optimize()

# 最优解结果输出
if model.status == GRB.OPTIMAL:
    print(f"最优解找到，总成本: {model.objVal:.2f}")

# 输出每个时隙的解
# 初始化存储结构
# deployment_results = {t: [] for t in time_slots}
# allocation_results = {t: [] for t in time_slots}
deployment_results = {}
allocation_results = {}

# ----------------------
# 提取容器部署状态 (y变量)
# ----------------------
for t in time_slots:
    # 当前时隙的服务器部署状态
    deployed_containers = np.zeros((server_num, container_num))
    for s_id, s in enumerate(servers):
        for c_id, c in enumerate(containers):
            server_id = s['server_id']
            deployed_containers[s_id][c_id] = (
                    y[c, server_id, t].X == 1  # 检查是否部署
            )
    deployment_results[t] = deployed_containers

# ----------------------
# 提取请求分配状态 (x变量)
# ----------------------
# 按时隙分组请求
from collections import defaultdict

slot_requests = defaultdict(list)
for r in requests:
    slot_requests[r['time_slot']].append(r)

# 遍历每个时隙的请求
for t in time_slots:
    # 遍历该时隙的所有请求
    current_allocation = np.zeros((len(slot_requests[t]), server_num + 1))
    for r_id, r in enumerate(slot_requests[t]):
        req_id = r['request_id']
        # 找到该请求的分配目标
        for k in x[req_id]:
            server_id = server_index_map[k]
            current_allocation[r_id][server_id] = (x[req_id][k].X > 0.5)
        # print(f'request_id: {req_id}, allocation: {current_allocation[r_id]}')
    allocation_results[t] = current_allocation


def save_to_yaml(deploy, alloc, filename):
    """将部署和分配结果保存为 pkl文件"""
    # 转换数据结构为适合 YAML 的格式
    yaml_data = {'deployment': deploy, 'allocation': alloc}
    # 写入文件
    with open(filename, 'wb') as f:
        pickle.dump(yaml_data, f)


# 使用示例
save_to_yaml(deployment_results,
             allocation_results,
             os.path.join(dataset_path, 'deployment_results.pkl'))

# =============================
# 8. 结果展示
# =============================
# def print_deployment(deployment_data):
#     """打印容器部署状态"""
#     print("\n容器部署状态:")
#     for t, servers in deployment_data.items():
#         print(f"\n时隙 {t}:")
#         for server_id, containers in servers.items():
#             if containers:
#                 print(f"  服务器 {server_id} 部署容器: {', '.join(containers)}")
#
#
# def print_allocation(allocation_data):
#     """打印请求分配状态"""
#     print("\n请求分配状态:")
#     for t, allocation in allocation_data.items():
#         print(f"\n时隙 {t}:")
#         # 边缘服务器分配
#         for server_id, req_ids in allocation['edge'].items():
#             print(f"  服务器 {server_id} 处理请求: {len(req_ids)}个")
#         # 云分配
#         print(f"  云端处理请求: {len(allocation['cloud'])}个")

# 打印结果
# print_deployment(deployment_results)
# print_allocation(allocation_results)

# =============================
# 9. 结果保存 (可选)
# =============================
# 保存到CSV
# import csv
#
# # 保存容器部署状态
# container_deployment_path = os.path.join(config.data_dir, 'container_deployment.csv')
# request_allocation_path = os.path.join(config.data_dir, 'request_allocation.csv')
# with open(container_deployment_path, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['time_slot', 'server_id', 'containers'])
#     for t in time_slots:
#         for server_id, containers in deployment_results[t].items():
#             writer.writerow([t, server_id, ','.join(containers)])
#
# # 保存请求分配状态
# with open(request_allocation_path, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['time_slot', 'server_id', 'request_count'])
#     for t in time_slots:
#         # 边缘服务器
#         for server_id, req_ids in allocation_results[t]['edge'].items():
#             writer.writerow([t, server_id, len(req_ids)])
#         # 云端
#         writer.writerow([t, 'cloud', len(allocation_results[t]['cloud'])])

# if model.solCount > 0:
#     # 输出服务器负载情况
#     for s in servers:
#         server_id = s['server_id']
#         print(f"\n服务器 {server_id} 资源利用率:")
#         for t in time_slots:
#             cpu_used = sum(r['cpu_demand'] * x[r['request_id']][server_id].X
#                            for r in requests if r['time_slot'] == t)
#             mem_used = sum(r['mem_demand'] * x[r['request_id']][server_id].X
#                            for r in requests if r['time_slot'] == t)
#             print(f"时隙 {t}: CPU={cpu_used:.1f}/{s['cpu_capacity']}核, MEM={mem_used:.1f}/{s['mem_capacity']}MB")
#
#     # 输出容器部署状态
#     print("\n关键容器部署:")
#     for c in containers:
#         for s in servers:
#             server_id = s['server_id']
#             deployed = any(y[c, server_id, t].X > 0.9 for t in time_slots)
#             if deployed:
#                 print(f"容器 {c} 部署在 {server_id}")
# else:
#     print("未找到可行解")
