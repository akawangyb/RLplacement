# --------------------------------------------------
# 文件名: create_container_request
# 创建时间: 2025/3/29 10:50
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
# 数据示例
# request_id	service_type	cpu_demand	mem_demand	upload_demand	download_demand	image_size	time_slot
# REQ_0001	video_transcode	2.35	1536	8	75	1.8	10
# REQ_0002	web_service	0.28	128	1	3	0.3	19
# ​服务类型	​业务场景	​内存保活大小 h_c (MB)	​设定依据
# ​web_service	HTTP服务、API网关	200~500	轻量级服务，依赖运行时（如Nginx、Node.js），内存占用较低。
# ​video_transcode	实时视频转码（H.264→H.265）	1024~2048	视频编解码需加载大型算法库（如FFmpeg），内存消耗高。
# ​database_query	分布式数据库查询（如MySQL、Redis）	512~1024	数据库连接池和缓存占用内存，但通常小于视频处理。
# ​ai_inference	深度学习推理（如ResNet、BERT）	2048~4096	预训练模型参数加载至内存（如PyTorch模型），内存需求极高。
# ​iot_analytics	物联网数据实时分析	256~512	流式数据处理框架（如Apache Flink）内存占用中等。
# ​Web服务：
# ​典型容器：Nginx（~50MB）、Node.js（~300MB）。
# ​内存分配：为容器运行时和请求处理预留额外空间，总占用200~500MB。
# ​视频转码：
# ​典型工具：FFmpeg加载H.265编码器需约1GB内存。
# ​动态扩展：4K视频转码可能占用更高内存，设置上限2048MB。
# ​数据库查询：
# ​连接池与缓存：MySQL容器默认配置约512MB，Redis缓存根据数据集大小调整。
# ​平衡策略：避免过度分配，设置512~1024MB。
# ​AI推理：
# ​模型参数：ResNet-50模型约200MB，但推理时需加载至内存并分配中间变量，总需求2~4GB。
# ​硬件适配：高端服务器（如GPU节点）通常配备大内存，支持高 h_c。
import numpy as np
import pandas as pd

import config

# =============================
# 生成容器请求数据集（24时隙）
# =============================
np.random.seed(42)

# 定义服务类型及其资源需求（单位：CPU核，内存MB，上行/下行带宽Mbps，镜像大小GB）

service_types = {
    "web_service": {
        "h_c": np.random.uniform(200, 500),  # 内存保活大小（MB）
        "compute_delay": (50, 200),
        "cpu": (0.1, 0.5),
        "mem": (64, 256),
        "upload": (0.5, 2),  # 上行（请求结果返回）
        "download": (1, 5),  # 下行（接收请求数据）
        "image_size": (0.2, 0.5)
    },
    "video_transcode": {
        "h_c": np.random.uniform(200, 2048),  # 内存保活大小（MB）
        "compute_delay": (500, 2000),
        "cpu": (1.0, 3.0),
        "mem": (512, 2048),
        "upload": (5, 10),  # 上行（转码后视频流）
        "download": (50, 100),  # 下行（原始视频流）
        "image_size": (1.0, 2.0)
    },
    "database_query": {
        "h_c": np.random.uniform(512, 1024),  # 内存保活大小（MB）
        "compute_delay": (10, 100),
        "cpu": (0.5, 2.0),
        "mem": (256, 1024),
        "upload": (10, 20),  # 上行（查询结果）
        "download": (1, 5),  # 下行（查询请求）
        "image_size": (0.5, 1.5)
    },
    "ai_inference": {
        "h_c": np.random.uniform(2048, 4096),  # 内存保活大小（MB）
        "compute_delay": (100, 300),
        "cpu": (2.0, 4.0),
        "mem": (1024, 4096),
        "upload": (20, 50),  # 上行（推理结果）
        "download": (50, 100),  # 下行（输入数据）
        "image_size": (2.0, 5.0)
    }
}

num_requests = config.num_requests
requests = []

# 时隙请求权重（高峰时段权重更高）
# time_slot_weights = [
#     0.2 if 0 <= h < 6 else  # 0-5时：低负载
#     0.5 if 6 <= h < 9 else  # 6-8时：早间负载
#     1.0 if 9 <= h < 18 else  # 9-17时：高峰时段
#     0.8 if 18 <= h < 22 else  # 18-21时：晚间负载
#     0.3  # 22-23时：深夜低负载
#     for h in range(24)
# ]
time_slot_weights = [1 / 24] * 24

# 分配请求到时隙
time_slots = np.random.choice(
    np.arange(24),
    size=num_requests,
    p=np.array(time_slot_weights) / sum(time_slot_weights)
)

# 根据4个生成10个容器
container_set = {}
for i in range(config.num_containers):
    service = np.random.choice(list(service_types.keys()))
    # service ="web_service"
    container_set[f"service{i + 1:04d}"] = {
        **service_types[service]
    }

print(container_set)
# 生成请求数据
for i in range(config.num_requests):
    service = np.random.choice(list(container_set.keys()))
    # service = np.random.choice(['database_query'])
    params = container_set[service]
    h_c = int(params["h_c"])
    compute_delay = int(np.random.uniform(*params["compute_delay"]))
    cpu = np.round(np.random.uniform(*params["cpu"]), 2)
    mem = int(np.random.uniform(*params["mem"]))
    upload = int(np.random.uniform(*params["upload"]))
    download = int(np.random.uniform(*params["download"]))
    image_size = np.round(np.random.uniform(*params["image_size"]), 2)
    slot = int(time_slots[i]) + 1  # 时隙1~24

    requests.append({
        "request_id": f"REQ_{i + 1:04d}",
        "service_type": service,
        "h_c": h_c,
        "compute_delay": compute_delay,
        "cpu_demand": cpu,
        "mem_demand": mem,
        "upload_demand": upload,
        "download_demand": download,
        "image_size": image_size,
        "time_slot": slot
    })
# print(requests[0])
#
# # 保存为CSV
df_requests = pd.DataFrame(requests)
df_requests.to_csv("container_requests.csv", index=False)
