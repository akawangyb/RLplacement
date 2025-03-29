# --------------------------------------------------
# 文件名: gurobi_rr
# 创建时间: 2025/3/29 15:13
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

edge_prop_delay = 50
cloud_prop_delay = 300
# =============================
# 1. 数据准备
# =============================

# 读取请求数据（已包含时隙信息）
df_requests = pd.read_csv("data/container_requests.csv")
requests = df_requests.to_dict('records')

# 读取服务器数据
df_servers = pd.read_csv("data/edge_servers.csv")
servers = df_servers.to_dict('records')

# 定义常量参数
time_slots = sorted(df_requests['time_slot'].unique())  # 1~24时隙
service_types = df_requests['service_type'].unique()
# 提取唯一的 service_type 和 h_c 组合
service_hc_map = df_requests[["service_type", "h_c"]].drop_duplicates().set_index("service_type")["h_c"].to_dict()
print(service_hc_map)

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
        ['cloud'] + [s['server_id'] for s in servers],
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        ub=1.0,
        name=f"x_{req_id}"
    )

# 镜像加载变量 y[c][n][t]
containers = list(df_requests['service_type'].unique())
y = model.addVars(
    containers,
    [s['server_id'] for s in servers],
    time_slots,
    vtype=GRB.CONTINUOUS,
    lb=0.0,
    ub=1.0,
    name="y"
)

# 镜像加载触发变量 z[c][n][t] (0-1变化时触发)
z = model.addVars(
    containers,
    [s['server_id'] for s in servers],
    time_slots,
    vtype=GRB.CONTINUOUS,
    lb=0.0,
    ub=1.0,
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
        alpha = 1.2  # 干扰因子
        # compute_delay = alpha * r['cpu_demand'] / s['cpu_capacity']
        # 不考虑干扰问题
        compute_delay = r["compute_delay"]
        # prop_delay = r['net_demand'] / s['net_capacity']
        total_cost += x[req_id][server_id] * (compute_delay + edge_prop_delay)

    # 云端计算惩罚
    # compute_delay_cloud = r['cpu_demand'] / 100  # 假设云端算力为100核
    compute_delay_cloud = r['compute_delay']
    total_cost += x[req_id]['cloud'] * (compute_delay_cloud + cloud_prop_delay)

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
        ) + gp.quicksum(
            y[c, server_id, t] * df_requests[df_requests['service_type'] == c]['image_size'].mean() * 1024  # MB
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
# 6. 求解与结果分析
# =============================
model.Params.TimeLimit = 60  # 10分钟限制
model.optimize()

# 最优解结果输出
if model.status == GRB.OPTIMAL:
    print(f"最优解找到，总成本: {model.objVal:.2f}")

if model.solCount > 0:
    # 输出服务器负载情况
    for s in servers:
        server_id = s['server_id']
        print(f"\n服务器 {server_id} 资源利用率:")
        for t in time_slots:
            cpu_used = sum(r['cpu_demand'] * x[r['request_id']][server_id].X
                           for r in requests if r['time_slot'] == t)
            mem_used = sum(r['mem_demand'] * x[r['request_id']][server_id].X
                           for r in requests if r['time_slot'] == t)
            print(f"时隙 {t}: CPU={cpu_used:.1f}/{s['cpu_capacity']}核, MEM={mem_used:.1f}/{s['mem_capacity']}MB")

    # 输出容器部署状态
    print("\n关键容器部署:")
    for c in containers:
        for s in servers:
            server_id = s['server_id']
            deployed = any(y[c, server_id, t].X > 0.9 for t in time_slots)
            if deployed:
                print(f"容器 {c} 部署在 {server_id}")
else:
    print("未找到可行解")

# 如何进行随机舍入
# 提取松弛解
x_relax = {}
for r in requests:
    req_id = r['request_id']
    x_relax[req_id] = {
        k: x[req_id][k].X
        for k in x[req_id]
    }

y_relax = {
    (c, n, t): y[c, n, t].X
    for c in containers
    for n in [s['server_id'] for s in servers]
    for t in time_slots
}

z_relax = {
    (c, n, t): z[c, n, t].X
    for c in containers
    for n in [s['server_id'] for s in servers]
    for t in time_slots
}
print(x_relax)
print(y_relax)
print(z_relax)


def round_x_request(x_relax_slot, requests_slot, servers):
    """修正后的舍入函数，保留完整分配字典"""
    x_rounded = {}
    candidates = ['cloud'] + [s['server_id'] for s in servers]

    for r in requests_slot:
        req_id = r['request_id']
        # 生成概率分布
        probs = [x_relax_slot[req_id][k] for k in candidates]
        probs = np.array(probs) / np.sum(probs)  # 归一化

        # 轮盘赌选择
        chosen = np.random.choice(candidates, p=probs)

        # 构建完整分配字典（被选中的位置为1，其余为0）
        x_rounded[req_id] = {k: 0 for k in candidates}
        x_rounded[req_id][chosen] = 1

    return x_rounded


def round_y_server(y_relax_slot, server_id, containers, h_c, S_mem):
    """舍入单个服务器单时隙的容器加载状态"""
    # 按松弛值降序排序
    sorted_containers = sorted(
        [(c, y_relax_slot[c]) for c in containers],
        key=lambda x: -x[1]
    )

    y_rounded = {}
    used_mem = 0

    # 贪心加载高概率容器
    for c, prob in sorted_containers:
        if used_mem + h_c[c] > S_mem[server_id]:
            y_rounded[c] = 0
            continue
        # 按概率随机决定是否加载
        if np.random.rand() <= prob:
            y_rounded[c] = 1
            used_mem += h_c[c]
        else:
            y_rounded[c] = 0
    return y_rounded


def compute_z(y_rounded_current, y_rounded_previous):
    """根据前后时隙的y值计算z"""
    z = {}
    for c in y_rounded_current:
        z[c] = 1 if y_rounded_current[c] == 1 and y_rounded_previous.get(c, 0) == 0 else 0
    return z


def get_all_slot():
    #  初始时隙服务其没有加载任何容器
    prev_y = {s['server_id']: {c: 0 for c in containers} for s in servers}
    # 存储最终解
    x_rounded_all = {t: {} for t in time_slots}
    y_rounded_all = {t: {} for t in time_slots}
    z_rounded_all = {t: {} for t in time_slots}
    for t in time_slots:
        # 获取当前时隙的请求
        requests_slot = [r for r in requests if r['time_slot'] == t]

        # 舍入x
        x_relax_slot = {req['request_id']: x_relax[req['request_id']] for req in requests_slot}
        x_rounded = round_x_request(x_relax_slot, requests_slot, servers)
        x_rounded_all[t] = x_rounded

        # 舍入y
        y_rounded_slot = {}
        for s in servers:
            server_id = s['server_id']
            # 提取当前服务器当前时隙的松弛解
            y_relax_server = {
                c: y_relax[c, server_id, t]
                for c in containers
            }
            # 执行舍入
            y_rounded = round_y_server(
                y_relax_slot=y_relax_server,
                server_id=server_id,
                containers=containers,
                h_c=service_hc_map,
                S_mem={s['server_id']: s['mem_capacity'] for s in servers}
            )
            y_rounded_slot[server_id] = y_rounded

        # 计算z
        z_rounded_slot = {}
        for s in servers:
            server_id = s['server_id']
            z_rounded = compute_z(
                y_rounded_current=y_rounded_slot[server_id],
                y_rounded_previous=prev_y[server_id]
            )
            z_rounded_slot[server_id] = z_rounded

        # 更新历史状态
        for s in servers:
            server_id = s['server_id']
            prev_y[server_id] = y_rounded_slot[server_id].copy()

        # 存储结果
        y_rounded_all[t] = y_rounded_slot
        z_rounded_all[t] = z_rounded_slot

    return x_rounded_all, y_rounded_all, z_rounded_all


def validate_solution(t, x_rounded, y_rounded, servers, requests):
    """验证单时隙解的可行性"""
    # 检查请求分配唯一性
    for r in requests:
        if r['time_slot'] != t:
            continue
        req_id = r['request_id']
        # 遍历所有候选位置（服务器+云）
        total = sum(x_rounded[req_id][k] for k in x_rounded[req_id])
        if total != 1:
            return False

    # 检查服务器资源约束
    for s in servers:
        server_id = s['server_id']
        # CPU
        cpu_used = sum(
            r['cpu_demand'] * x_rounded[r['request_id']][server_id]
            for r in requests if r['time_slot'] == t
        )
        if cpu_used > s['cpu_capacity']:
            return False

        # 内存
        mem_used = sum(
            r['mem_demand'] * x_rounded[r['request_id']][server_id]
            for r in requests if r['time_slot'] == t
        ) + sum(
            y_rounded[server_id][c] * service_hc_map[c]
            for c in containers
        )
        if mem_used > s['mem_capacity']:
            return False
    return True


# 验证所有时隙
feasible = True

while feasible:
    x_rounded_all, y_rounded_all, z_rounded_all = get_all_slot()
    for t in time_slots:
        if not validate_solution(
                t,
                x_rounded_all[t],
                y_rounded_all[t],
                servers,
                [r for r in requests if r['time_slot'] == t]
        ):
            print(f"时隙 {t} 的解不可行！")
            feasible = False
            break
    if not feasible:
        feasible = True

if feasible:
    print("所有时隙的解均可行！")
else:
    print("存在不可行时隙，需进行修复")
