# --------------------------------------------------
# 文件名: create_server_data
# 创建时间: 2025/3/29 10:51
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
# 数据示例
# server_id	tier	cpu_capacity	mem_capacity	upload_capacity	download_capacity	location
# EDGE_001	mid-tier	12	24576	750	1500	Zone-3
# EDGE_002	high-tier	24	65536	1500	3500	Zone-1
# =============================
# 生成边缘服务器参数
# =============================
import numpy as np
import pandas as pd

import config

np.random.seed(42)
# 定义服务器加载速度基准（MB/s）
base_load_speed = 10  # 基准加载速度

server_specs = {
    "low-tier": {
        "b_n": base_load_speed * 0.5,  # 低端服务器加载速度较慢
        "cpu": (4, 8),
        "mem": (8, 16),
        "storage": (1, 2),
        "upload": (100, 500),  # 上行带宽（Mbps）
        "download": (500, 1000)  # 下行带宽（Mbps）
    },
    "mid-tier": {
        "b_n": base_load_speed * 1.0,
        "cpu": (8, 16),
        "mem": (16, 32),
        "storage": (2, 4),
        "upload": (500, 1000),
        "download": (1000, 2000)
    },
    "high-tier": {
        "b_n": base_load_speed * 2.0,
        "cpu": (16, 32),
        "mem": (32, 64),
        "storage": (4, 8),
        "upload": (1000, 2000),
        "download": (2000, 5000)
    }
}

num_servers = config.num_servers
servers = []

for _ in range(num_servers):
    tier = np.random.choice(list(server_specs.keys()), p=[0, 0, 1])
    params = server_specs[tier]

    cpu = int(np.random.uniform(*params["cpu"]))
    mem = int(np.random.uniform(*params["mem"])) * 1024  # 转换为MB
    storage = int(np.random.uniform(*params["storage"]))
    upload = int(np.random.uniform(*params["upload"]))
    download = int(np.random.uniform(*params["download"]))

    servers.append({
        "server_id": f"EDGE_{len(servers) + 1:03d}",
        "tier": tier,
        "b_n": params["b_n"],  # 加载速度（MB/s）
        "cpu_capacity": cpu,
        "mem_capacity": mem,
        "storage_capacity": storage,
        "upload_capacity": upload,
        "download_capacity": download,
        "location": f"Zone-{np.random.randint(1, 6)}"
    })

# 保存为CSV
df_servers = pd.DataFrame(servers)
df_servers.to_csv("edge_servers.csv", index=False)
