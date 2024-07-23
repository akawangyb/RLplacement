# --------------------------------------------------
# 文件名: process
# 创建时间: 2024/7/19 22:54
# 描述: 处理原始数据集
# 作者: WangYuanbo
# --------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset/dataset.csv')
print(df.head())

# 计算新的cpu列
df['cpu'] = df['cpu_demand'] / df['cpu_supply']
df['mem'] = df['mem_demand'] / df['mem_supply']
df['net_in'] = df['net_in_demand'] / df['net_in_supply']
df['net_out'] = df['net_out_demand'] / df['net_out_supply']
df['read'] = df['read_demand'] / df['read_supply']
df['write'] = df['write_demand'] / df['write_supply']

# 删除demand和supply列
df.drop(
    columns=['cpu_demand', 'mem_demand', 'net_in_demand', 'net_out_demand', 'read_demand', 'write_demand', 'cpu_supply',
             'mem_supply', 'net_in_supply', 'net_out_supply', 'read_supply', 'write_supply'], inplace=True)

# 重新排序列
df = df[['cpu', 'mem', 'net_in', 'net_out', 'read', 'write', 'target']]

# 显示DataFrame
print(df)
df.to_csv('dataset_new.csv', index=False)

# 拆分为训练集、测试集和验证集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)

# 显示拆分后的数据集大小
print("训练集大小：", train_df.shape)
print("测试集大小：", test_df.shape)
print("验证集大小：", valid_df.shape)

# 将数据集写入新文件
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
valid_df.to_csv('valid.csv', index=False)
