# --------------------------------------------------
# 文件名: gurobi_slot_rr
# 创建时间: 2025/3/29 16:43
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import os.path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

edge_prop_delay = 50
cloud_prop_delay = 300
# =============================
# 1. 数据准备
# =============================

dataset_path=r"data/test"
# 读取请求数据（已包含时隙信息）
df_requests = os.path.join(dataset_path, 'container_requests.csv')
df_requests=pd.read_csv(df_requests)
requests = df_requests.to_dict('records')

# 读取服务器数据
df_servers = os.path.join(dataset_path, 'edge_servers.csv')
df_servers = pd.read_csv(df_servers)
servers = df_servers.to_dict('records')

# 定义常量参数
time_slots = sorted(df_requests['time_slot'].unique())  # 1~24时隙
service_types = df_requests['service_type'].unique()
# 提取唯一的 service_type 和 h_c 组合
service_hc_map = df_requests[["service_type", "h_c"]].drop_duplicates().set_index("service_type")["h_c"].to_dict()
print(service_hc_map)

server_mem_capacity = {
    s['server_id']: s['mem_capacity']
    for s in df_servers.to_dict('records')
}
# 构建服务器资源字典
server_resources = {
    s['server_id']: {
        'cpu': s['cpu_capacity'],
        'mem': s['mem_capacity'],
        'upload': s['upload_capacity'],
        'download': s['download_capacity']
    }
    for s in servers
}

containers = list(df_requests['service_type'].unique())


def gurobi_slove():
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
            # 不考虑干扰问题
            compute_delay = r["compute_delay"]
            total_cost += x[req_id][server_id] * (compute_delay + edge_prop_delay)

        # 云端计算惩罚
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

    return model


def validate_action(t, x_rounded, y_rounded, servers, requests_t):
    """校验并修复当前时隙解"""
    # 校验请求分配唯一性
    for req in requests_t:
        if x_rounded[req['request_id']] not in ['cloud'] + [s['server_id'] for s in servers]:
            return False

    # 校验服务器资源
    for server in servers:
        s_id = server['server_id']
        cpu_used = sum(req['cpu_demand'] for req in requests_t if x_rounded[req['request_id']] == s_id)
        upload_used = sum(req['upload_demand'] for req in requests_t if x_rounded[req['request_id']] == s_id)
        download_used = sum(req['download_demand'] for req in requests_t if x_rounded[req['request_id']] == s_id)
        mem_used = sum(req['mem_demand'] for req in requests_t if x_rounded[req['request_id']] == s_id) + \
                   sum(service_hc_map[c] * y_rounded[s_id][c] for c in containers)
        if (cpu_used > server['cpu_capacity'] or
                mem_used > server['mem_capacity'] or
                upload_used > server['upload_capacity'] or
                download_used > server['download_capacity']
        ):
            return False
    return True


def round_containers(y_relax_t, prev_y, h_c, S_mem, servers):
    # y_rounded是一个二维字典
    """舍入当前时隙容器加载状态"""
    y_rounded = {}
    for server in servers:
        server_id = server['server_id']
        # 按松弛值降序排序容器
        sorted_containers = sorted(
            [(c, y_relax_t[(c, server_id)]) for c in containers],
            key=lambda x: -x[1]
        )
        # 贪心加载
        loaded = []
        used_mem = 0
        for c, prob in sorted_containers:
            if used_mem + h_c[c] > server['mem_capacity']:
                continue
            # 按概率随机加载
            if np.random.rand() <= prob:
                loaded.append(c)
                used_mem += h_c[c]
        # 记录结果
        y_rounded[server_id] = {c: 1 if c in loaded else 0 for c in containers}
    return y_rounded


def compute_mirror_loading(y_rounded_t, prev_y):
    """计算镜像加载动作 z"""
    z_rounded = {}
    for server_id in y_rounded_t:
        z_rounded[server_id] = {}
        for c in containers:
            # z=1 当且仅当 y_t=1 且 y_{t-1}=0
            current = y_rounded_t[server_id][c]
            previous = prev_y[server_id].get(c, 0)
            z_rounded[server_id][c] = 1 if (current == 1 and previous == 0) else 0
    return z_rounded


def round_requests(x_relax_t, requests_t, servers, S_resources):
    """舍入当前时隙请求分配"""
    x_rounded = {}
    resource_usage = {s['server_id']: {'cpu': 0, 'mem': 0, 'upload': 0, 'download': 0} for s in servers}

    for req in requests_t:
        req_id = req['request_id']
        candidates = ['cloud'] + [s['server_id'] for s in servers]
        # 生成归一化概率
        probs = [x_relax_t[req_id][k] for k in candidates]
        probs = np.array(probs) / np.sum(probs)
        # 轮盘赌选择初始目标
        target = np.random.choice(candidates, p=probs)

        # 资源冲突处理
        if target != 'cloud':
            s = next(s for s in servers if s['server_id'] == target)
            # 检查资源是否超限
            cpu_new = resource_usage[target]['cpu'] + req['cpu_demand']
            mem_new = resource_usage[target]['mem'] + req['mem_demand']
            upload_new = resource_usage[target]['upload'] + req['upload_demand']
            download_new = resource_usage[target]['download'] + req['download_demand']
            if (cpu_new > s['cpu_capacity'] or
                    mem_new > s['mem_capacity'] or
                    upload_new > s['upload_capacity'] or
                    download_new > s['download_capacity']):
                target = 'cloud'  # 回退到云

        # 记录分配结果
        x_rounded[req_id] = target
        if target != 'cloud':
            resource_usage[target]['cpu'] += req['cpu_demand']
            resource_usage[target]['mem'] += req['mem_demand']
            resource_usage[target]['download'] += req['download_demand']
            resource_usage[target]['upload'] += req['upload_demand']

    return x_rounded, resource_usage


def get_y_relax_from_model(model, t):
    """
    从Gurobi模型中提取指定时隙t的容器加载状态松弛解（y变量）
    参数:
        model: Gurobi模型对象
        t: 目标时隙（整数）
    返回:
        y_relax: 字典，键为 (容器类型, 服务器ID)，值为松弛解值（浮点数）
    """
    y_relax = {}
    # 遍历所有容器和服务器
    for c in containers:
        for s in servers:
            server_id = s['server_id']
            # 构造Gurobi变量名（与模型定义一致）
            var_name = f"y[{c},{server_id},{t}]"
            # 获取变量对象
            var = model.getVarByName(var_name)
            if var is not None:
                y_relax[(c, server_id)] = var.X
            else:
                raise ValueError(f"变量 {var_name} 未在模型中找到！")
    return y_relax


def get_x_relax_from_model(model, t):
    """提取指定时隙t的请求分配松弛解（x变量）"""
    x_relax = {}
    # 遍历当前时隙的所有请求
    for r in requests:
        if r['time_slot'] != t:
            continue
        req_id = r['request_id']
        # 构造候选位置列表（服务器ID + 'cloud'）
        candidates = ['cloud'] + [s['server_id'] for s in servers]
        # 提取每个候选位置的松弛值
        x_relax[req_id] = {}
        for k in candidates:
            var_name = f"x_{req_id}[{k}]"
            var = model.getVarByName(var_name)
            x_relax[req_id][k] = var.X if var else 0.0
    return x_relax


# 初始化历史状态
prev_y = {s['server_id']: {c: 0 for c in containers} for s in servers}

# 存储所有时隙结果
all_x = {t: {} for t in time_slots}
all_y = {t: {} for t in time_slots}
all_z = {t: {} for t in time_slots}
model = gurobi_slove()

for t in time_slots:
    # 1. 舍入容器加载状态
    y_relax_t = get_y_relax_from_model(model=model, t=t)  # 从松弛解中提取当前时隙的y_relax
    y_rounded = round_containers(y_relax_t, prev_y, service_hc_map, server_mem_capacity, servers)
    all_y[t] = y_rounded

    # 2. 计算镜像加载动作
    z_rounded = compute_mirror_loading(y_rounded, prev_y)
    all_z[t] = z_rounded

    # 3. 舍入请求分配
    x_relax_t = get_x_relax_from_model(model=model, t=t)  # 提取当前时隙的x_relax
    requests_t = [r for r in requests if r['time_slot'] == t]
    x_rounded, resource_usage = round_requests(x_relax_t, requests_t, servers, S_resources=server_resources)
    all_x[t] = x_rounded
    if validate_action(t, x_rounded, y_rounded=y_rounded, servers=servers, requests_t=requests_t):
        print(t, '舍入成功')
    # 4. 更新历史状态
    prev_y = y_rounded.copy()


def compute_action_overall_delay(all_x, all_y, all_z):
    """
    求出松弛解在每一个时隙的时间
    :param all_x: 每个时隙的x的动作
    :param all_y: 每个时隙y的动作
    :param all_z: 每个时隙z的动作
    """
    print(all_x[1])
    overall_delay = 0
    for t in time_slots:
        x_action = all_x[t]
        y_action = all_y[t]
        z_action = all_z[t]
        # 计算3个实验
        # 启动时延
        # 镜像加载时延成本
        # 定义目标函数中的镜像加载时延项
        loading_delay = 0
        for s in servers:
            server_id = s['server_id']
            b_n = s['b_n']  # 加载速度（MB/s）
            for c in containers:
                h_c = service_hc_map[c]  # 容器保活内存（MB）
                loading_delay += (h_c / b_n) * z_action[server_id][c]

        computing_delay = 0
        # 计算时延和传播时延
        for r in requests:
            req_id = r['request_id']
            c = r['service_type']
            ts = r['time_slot']
            if ts != t:
                continue
            # 边缘计算时延
            if req_id not in x_action:
                continue
            for s in servers:
                server_id = s['server_id']
                if server_id not in x_action[req_id]:
                    continue
                # 不考虑干扰问题
                compute_delay = r["compute_delay"]
                computing_delay += compute_delay + edge_prop_delay

            # 云端计算惩罚
            compute_delay_cloud = r['compute_delay']
            computing_delay += (compute_delay_cloud + cloud_prop_delay)
        overall_delay += computing_delay + loading_delay
        print('computing_delay', computing_delay)
        print('loading_delay', loading_delay)
        print('ts', t, 'total delay', loading_delay + computing_delay)
    print(overall_delay)


def validate_and_repair(t, x_rounded, y_rounded, servers, requests_t):
    """校验并修复当前时隙解"""
    # 校验请求分配唯一性
    for req in requests_t:
        if x_rounded[req['request_id']] not in ['cloud'] + [s['server_id'] for s in servers]:
            return False

    # 校验服务器资源
    for server in servers:
        s_id = server['server_id']
        cpu_used = sum(req['cpu_demand'] for req in requests_t if x_rounded[req['request_id']] == s_id)
        mem_used = sum(req['mem_demand'] for req in requests_t if x_rounded[req['request_id']] == s_id) + \
                   sum(service_hc_map[c] * y_rounded[s_id][c] for c in containers)
        if cpu_used > server['cpu_capacity'] or mem_used > server['mem_capacity']:
            # 修复策略：随机迁移部分请求到云
            problematic_reqs = [req for req in requests_t if x_rounded[req['request_id']] == s_id]
            np.random.shuffle(problematic_reqs)
            for req in problematic_reqs:
                x_rounded[req['request_id']] = 'cloud'
                cpu_used -= req['cpu_demand']
                mem_used -= req['mem_demand']
                if cpu_used <= server['cpu_capacity'] and mem_used <= server['mem_capacity']:
                    break
    return True


if __name__ == '__main__':
    compute_action_overall_delay(all_x, all_y, all_z)
