# --------------------------------------------------
# 文件名: linearregression
# 创建时间: 2024/7/19 23:20
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset/dataset.csv')

df['cpu_demand'] /= 56
df['cpu_supply'] /= 56
df['mem_demand'] /= 256
df['mem_supply'] /= 256
df['net_in_demand'] /= 1000
df['net_in_supply'] /= 1000
df['net_out_demand'] /= 1000
df['net_out_supply'] /= 1000
df['read_demand'] /= 200
df['read_supply'] /= 200
df['write_demand'] /= 200
df['write_supply'] /= 200
# 拆分为训练集、测试集和验证集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)

# 提取特征和目标变量
train_features = train_df.drop('target', axis=1)
train_target = train_df['target']
test_features = test_df.drop('target', axis=1)
test_target = test_df['target']

# 创建线性回归模型并进行训练
model = LinearRegression()
model.fit(train_features, train_target)

# 进行预测
train_predictions = model.predict(train_features)
test_predictions = model.predict(test_features)

# 评估模型性能
train_mse = mean_squared_error(train_target, train_predictions)
train_r2 = r2_score(train_target, train_predictions)
test_mse = mean_squared_error(test_target, test_predictions)
test_r2 = r2_score(test_target, test_predictions)

# 打印评估结果
print("训练集均方误差(MSE)：", train_mse)
print("训练集R2分数：", train_r2)
print("测试集均方误差(MSE)：", test_mse)
print("测试集R2分数：", test_r2)

# # 保存模型
# # 将它们保存到字典中
weights = model.coef_
bias = model.intercept_
model_parameters = {"weights": weights, "bias": bias}

# 然后可以用pickle库来保存这个字典
with open('model_parameters.pkl', 'wb') as f:
    pickle.dump(model_parameters, f)
