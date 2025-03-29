# --------------------------------------------------
# 文件名: gurobi_rr
# 创建时间: 2025/3/29 15:13
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import gurobipy as gp
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
    # 获取所有变量值
all_vars = model.getVars()
var_values = {var.varName: var.X for var in all_vars}

# 打印结果
print("\n所有变量值:")
for name, value in var_values.items():
    print(f"{name} = {value:.2f}")

# 按名称获取单个变量
z_0_1 = model.getVarByName("z[0,1]")
print("\nz[0,1]的值:", z_0_1.X)
