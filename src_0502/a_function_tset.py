# --------------------------------------------------
# 文件名: a_function_tset
# 创建时间: 2024/5/8 22:17
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import torch

# 假设action是你的输入张量
action = torch.tensor([1, 2, 3],dtype=torch.float32)

# 使用randn_like，生成一个和action形状相同，元素服从标准正态分布的张量
new_tensor = torch.randn_like(action)

print(new_tensor)