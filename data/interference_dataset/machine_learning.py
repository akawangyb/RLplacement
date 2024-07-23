# --------------------------------------------------
# 文件名: machine_learning
# 创建时间: 2024/7/19 23:09
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------

import time

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

train_data_path = 'dataset/train.csv'
valid_data_path = 'dataset/valid.csv'
test_data_path = 'dataset/test.csv'

X_train = pd.read_csv(train_data_path)
col = [e for e in X_train.columns if e != 'target']

y_train = X_train['target']
X_train = X_train[col]

X_test = pd.read_csv(test_data_path)
y_test = X_test['target']
X_test = X_test[col]

X_val = pd.read_csv(valid_data_path)
y_val = X_val['target']
X_val = X_val[col]

# 创建CatBoost回归模型, epochs只设置了50轮为了快速示范, 在实际问题中可能需要更多轮数
# model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6)
#
# 训练模型, 你可能希望在这步设置更多参数, 例如学习率，深度等
# model.fit(X_train, y_train)
model = CatBoostRegressor()
model.load_model('catboost_model.bin')
# 进行预测
# 预测开始时间
start_time = time.time()
y_pred = model.predict(X_test)
# 预测结束时间
end_time = time.time()
test_mse = mean_squared_error(y_test, y_pred)
# model.save_model("catboost_model.bin")
print("Test MSE:", test_mse)
vector = [0.9, 0.4, 0.1, 0.1, 0, 0]
res = model.predict(vector)
print(res)
vector = [0.5, 0.4, 0.5, 0.1, 0, 0]
res = model.predict(vector)
print(res)
# 计算预测时长
prediction_time = end_time - start_time
# 计算每个预测数据的平均时长
average_prediction_time = prediction_time / len(X_test)
print("预测时长：", prediction_time)
print("每个测试数据的平均预测时长：", average_prediction_time)
# 计算MSE
# 保存模型权重
model.save_model('catboost_model.bin')
