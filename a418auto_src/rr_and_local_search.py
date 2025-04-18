# --------------------------------------------------
# 文件名: rr_and_local_search
# 创建时间: 2025/4/3 10:33
# 描述: 基于随机舍入技术求解，然后在进行局部搜索调整
# 作者: WangYuanbo
# --------------------------------------------------

import gurobipy as gp
import numpy as np
from gurobipy import GRB

edge_prop_delay = 50
cloud_prop_delay = 300


# =============================
# 1. 数据准备
# =============================

# dataset_path = r"data/test"
# # 读取请求数据（已包含时隙信息）
# df_requests = os.path.join(dataset_path, 'container_requests.csv')
# df_requests = pd.read_csv(df_requests)
# # requests = df_requests.to_dict('records')
#
# # 读取服务器数据
# df_servers = os.path.join(dataset_path, 'edge_servers.csv')
# df_servers = pd.read_csv(df_servers)
# # servers = df_servers.to_dict('records')
#
# # 定义常量参数
# time_slots = sorted(df_requests['time_slot'].unique())  # 1~24时隙
# service_types = df_requests['service_type'].unique()
# # 提取唯一的 service_type 和 h_c 组合
# service_hc_map = df_requests[["service_type", "h_c"]].drop_duplicates().set_index("service_type")["h_c"].to_dict()


# server_mem_capacity = {
#     s['server_id']: s['mem_capacity']
#     for s in df_servers.to_dict('records')
# }
# 构建服务器资源字典
# server_resources = {
#     s['server_id']: {
#         'cpu': s['cpu_capacity'],
#         'mem': s['mem_capacity'],
#         'upload': s['upload_capacity'],
#         'download': s['download_capacity']
#     }
#     for s in servers
# }

# containers = list(df_requests['service_type'].unique())


# 你回忆一下刚才我提出的两时间尺度的容器部署与请求分发问题
# 如果在已知容器部署决策x的情况，在不考虑性能干扰的情况下，如何使用gurobi求出请求分发决策y，
def gurobi_solve_not_relax(y: np.ndarray, requests, servers, containers, type_map, h_c_map):
    # =============================
    # 2. 创建Gurobi模型
    # =============================
    model = gp.Model("EdgeContainerDeployment")
    model.setParam('OutputFlag', 0)  # 关闭所有输出
    # =============================
    # 定义决策变量
    # =============================
    # 请求分配变量 x[r][k]
    x = {}
    for r_id, r in enumerate(requests):
        x[r_id] = model.addVars(
            range(len(servers) + 1),
            vtype=GRB.BINARY,
            name=f"x_{r_id}"
        )
    # =============================
    # 4. 定义目标函数
    # =============================
    total_cost = gp.QuadExpr()
    # 计算时延和传播时延
    for r_id, r in enumerate(requests):
        # 边缘计算时延
        for server_id, server in enumerate(servers):
            # 不考虑干扰问题
            compute_delay = r["compute_delay"]
            total_cost += x[r_id][server_id] * (compute_delay + edge_prop_delay)
        # 云端计算惩罚
        compute_delay_cloud = r['compute_delay']
        total_cost += x[r_id][len(servers)] * (compute_delay_cloud + cloud_prop_delay)
    model.setObjective(total_cost, GRB.MINIMIZE)
    # =============================
    # 5. 定义约束条件
    # =============================
    # 约束1：每个请求必须分配到唯一位置
    for r_id, r in enumerate(requests):
        model.addConstr(
            gp.quicksum(x[r_id][k] for k in range(len(servers) + 1)) == 1,
            name=f"uniq_alloc_{r_id}"
        )

    # 约束2：资源容量限制
    for server_id, s in enumerate(servers):
        # 获取当前时隙的请求
        slot_requests = requests
        # 内存约束 动态内存
        mem_usage = gp.quicksum(
            r['mem_demand'] * x[r_id][server_id]
            for r_id, r in enumerate(slot_requests)
        )
        # 镜像内存
        mem_usage += gp.quicksum(
            y[server_id][c_id] * h_c_map[c]
            for c_id, c in enumerate(containers)
        )
        model.addConstr(mem_usage <= s['mem_capacity'], name=f"mem_{server_id}")

        # CPU约束
        cpu_usage = gp.quicksum(
            r['cpu_demand'] * x[r_id][server_id]
            for r_id, r in enumerate(slot_requests)
        )
        model.addConstr(cpu_usage <= s['cpu_capacity'], name=f"cpu_{server_id}")

        # 服务器上行带宽约束
        model.addConstr(
            gp.quicksum(
                r['upload_demand'] * x[r_id][server_id]
                for r_id, r in enumerate(slot_requests)
            ) <= s['upload_capacity'],
            name=f"upload_{server_id}"
        )

        # 服务器下行带宽约束
        model.addConstr(
            gp.quicksum(
                r['download_demand'] * x[r_id][server_id]
                for r_id, r in enumerate(slot_requests)
            ) <= s['download_capacity'],
            name=f"download_{server_id}"
        )

    # 约束3：镜像加载依赖
    for r_id, r in enumerate(requests):
        c = type_map[r['service_type']]
        for server_id, s in enumerate(servers):
            model.addConstr(
                x[r_id][server_id] <= y[server_id][c],
                name=f"img_dep_{r_id}_{server_id}"
            )

    # =============================
    # 6. 求解与结果分析
    # =============================
    model.Params.TimeLimit = 60  # 10分钟限制
    model.optimize()
    # 把松弛解从模型中提取出来
    # 把解从模型中提取出来
    x_res = np.zeros((len(requests), len(servers) + 1))
    # 遍历当前时隙的所有请求
    for r_id, r in enumerate(requests):
        # 构造候选位置列表（服务器ID + 'cloud'）
        candidates = range(len(servers) + 1)
        # 提取每个候选位置的值
        for k in candidates:
            var_name = f"x_{r_id}[{k}]"
            var = model.getVarByName(var_name)
            x_res[r_id][k] = var.X if var else 0

    return x_res


def gurobi_solve_relax(y: np.ndarray, requests, servers, containers, type_map, h_c_map):
    """
    :param y: 容器的加载动作
    :return: 根据y求出一个请求分发决策
    """
    # =============================
    # 2. 创建Gurobi模型
    # =============================
    model = gp.Model("EdgeContainerDeployment")
    model.setParam('OutputFlag', 0)  # 关闭所有输出
    # =============================
    # 定义决策变量
    # =============================
    # 请求分配变量 x[r][k]
    x = {}
    for r_id, r in enumerate(requests):
        x[r_id] = model.addVars(
            range(len(servers) + 1),
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=1.0,
            name=f"x_{r_id}"
        )
    # =============================
    # 4. 定义目标函数
    # =============================
    total_cost = gp.QuadExpr()
    # 计算时延和传播时延
    for r_id, r in enumerate(requests):
        # 边缘计算时延
        for server_id, server in enumerate(servers):
            # 不考虑干扰问题
            compute_delay = r["compute_delay"]
            total_cost += x[r_id][server_id] * (compute_delay + edge_prop_delay)
        # 云端计算惩罚
        compute_delay_cloud = r['compute_delay']
        total_cost += x[r_id][len(servers)] * (compute_delay_cloud + cloud_prop_delay)
    model.setObjective(total_cost, GRB.MINIMIZE)
    # =============================
    # 5. 定义约束条件
    # =============================
    # 约束1：每个请求必须分配到唯一位置
    for r_id, r in enumerate(requests):
        model.addConstr(
            gp.quicksum(x[r_id][k] for k in range(len(servers) + 1)) == 1,
            name=f"uniq_alloc_{r_id}"
        )

    # 约束2：资源容量限制
    for server_id, s in enumerate(servers):
        # 获取当前时隙的请求
        slot_requests = requests
        # 内存约束 动态内存
        mem_usage = gp.quicksum(
            r['mem_demand'] * x[r_id][server_id]
            for r_id, r in enumerate(slot_requests)
        )
        # 镜像内存
        mem_usage += gp.quicksum(
            y[server_id][c_id] * h_c_map[c]
            for c_id, c in enumerate(containers)
        )
        # mem_usage += gp.quicksum(
        #     y[server_id][type_map[r['service_type']]] * r['h_c']
        #     for r in slot_requests
        # )
        model.addConstr(mem_usage <= s['mem_capacity'], name=f"mem_{server_id}")

        # CPU约束
        cpu_usage = gp.quicksum(
            r['cpu_demand'] * x[r_id][server_id]
            for r_id, r in enumerate(slot_requests)
        )
        model.addConstr(cpu_usage <= s['cpu_capacity'], name=f"cpu_{server_id}")

        # 服务器上行带宽约束
        model.addConstr(
            gp.quicksum(
                r['upload_demand'] * x[r_id][server_id]
                for r_id, r in enumerate(slot_requests)
            ) <= s['upload_capacity'],
            name=f"upload_{server_id}"
        )

        # 服务器下行带宽约束
        model.addConstr(
            gp.quicksum(
                r['download_demand'] * x[r_id][server_id]
                for r_id, r in enumerate(slot_requests)
            ) <= s['download_capacity'],
            name=f"download_{server_id}"
        )

    # 约束3：镜像加载依赖
    for r_id, r in enumerate(requests):
        c = type_map[r['service_type']]
        for server_id, s in enumerate(servers):
            model.addConstr(
                x[r_id][server_id] <= y[server_id][c],
                name=f"img_dep_{r_id}_{server_id}"
            )

    # =============================
    # 6. 求解与结果分析
    # =============================
    model.Params.TimeLimit = 60  # 10分钟限制
    model.optimize()
    # 把松弛解从模型中提取出来
    # x[rs_id][server_id]
    x = get_x_relax_from_model(model=model, requests=requests, servers=servers)
    # 对x进行随机舍入

    return x


# def validate_action(x_rounded, servers, requests_t, y, containers):
#     """校验并修复当前时隙解"""
#     # 校验请求分配唯一性
#     for req in requests_t:
#         if x_rounded[req['request_id']] not in ['cloud'] + [s['server_id'] for s in servers]:
#             return False
#
#     # 校验服务器资源
#     for server in servers:
#         s_id = server['server_id']
#         cpu_used = sum(req['cpu_demand'] for req in requests_t if x_rounded[req['request_id']] == s_id)
#         upload_used = sum(req['upload_demand'] for req in requests_t if x_rounded[req['request_id']] == s_id)
#         download_used = sum(req['download_demand'] for req in requests_t if x_rounded[req['request_id']] == s_id)
#         mem_used = sum(req['mem_demand'] for req in requests_t if x_rounded[req['request_id']] == s_id) + \
#                    sum(service_hc_map[c] * y[s_id][c] for c in containers)
#         if (cpu_used > server['cpu_capacity'] or
#                 mem_used > server['mem_capacity'] or
#                 upload_used > server['upload_capacity'] or
#                 download_used > server['download_capacity']
#         ):
#             return False
#     return True


def round_requests(x_relax_t, requests_t, servers, ):
    """舍入当前时隙请求分配"""
    x_rounded = []
    resource_usage = {s['server_id']: {'cpu': 0, 'mem': 0, 'upload': 0, 'download': 0} for s in servers}
    while True:
        tag = False
        for req in requests_t:
            req_id = req['request_id']
            candidates = ['cloud'] + [s['server_id'] for s in servers]
            # 生成归一化概率
            probs = [x_relax_t[req_id][k] for k in candidates]
            probs = np.array(probs) / np.sum(probs)
            # 轮盘赌选择初始目标
            target = np.random.choice(candidates, p=probs)
            x_rounded[req_id] = target
            # 资源冲突处理
            if target != 'cloud':
                # s = next(s for s in servers if s['server_id'] == target)
                s = servers[target]
                # 检查资源是否超限
                cpu_new = resource_usage[target]['cpu'] + req['cpu_demand']
                mem_new = resource_usage[target]['mem'] + req['mem_demand']
                upload_new = resource_usage[target]['upload'] + req['upload_demand']
                download_new = resource_usage[target]['download'] + req['download_demand']
                if (cpu_new > s['cpu_capacity'] or
                        mem_new > s['mem_capacity'] or
                        upload_new > s['upload_capacity'] or
                        download_new > s['download_capacity']):
                    # target = 'cloud'  # 回退到云
                    # 重新随机舍入
                    tag = True
                    break
            if tag:
                break
        if not tag:
            break
    return x_rounded


def get_x_relax_from_model(model, requests, servers):
    """提取指定时隙t的请求分配松弛解（x变量）"""
    x_relax = [[0] * (len(servers) + 1)] * len(requests)
    # 遍历当前时隙的所有请求
    for r_id, r in enumerate(requests):
        # 构造候选位置列表（服务器ID + 'cloud'）
        candidates = range(len(servers) + 1)
        # 提取每个候选位置的松弛值
        for k in candidates:
            var_name = f"x_{r_id}[{k}]"
            var = model.getVarByName(var_name)
            x_relax[r_id][k] = var.X if var else 0.0
    return x_relax


# def validate_and_repair(t, x_rounded, y_rounded, servers, requests_t, containers):
#     """校验并修复当前时隙解"""
#     # 校验请求分配唯一性
#     for req in requests_t:
#         if x_rounded[req['request_id']] not in ['cloud'] + [s['server_id'] for s in servers]:
#             return False
#
#     # 校验服务器资源
#     for server in servers:
#         s_id = server['server_id']
#         cpu_used = sum(req['cpu_demand'] for req in requests_t if x_rounded[req['request_id']] == s_id)
#         mem_used = sum(req['mem_demand'] for req in requests_t if x_rounded[req['request_id']] == s_id) + \
#                    sum(service_hc_map[c] * y_rounded[s_id][c] for c in containers)
#         if cpu_used > server['cpu_capacity'] or mem_used > server['mem_capacity']:
#             # 修复策略：随机迁移部分请求到云
#             problematic_reqs = [req for req in requests_t if x_rounded[req['request_id']] == s_id]
#             np.random.shuffle(problematic_reqs)
#             for req in problematic_reqs:
#                 x_rounded[req['request_id']] = 'cloud'
#                 cpu_used -= req['cpu_demand']
#                 mem_used -= req['mem_demand']
#                 if cpu_used <= server['cpu_capacity'] and mem_used <= server['mem_capacity']:
#                     break
#     return True




if __name__ == '__main__':
    # compute_action_overall_delay(all_x, all_y, all_z)
    print('hello')
