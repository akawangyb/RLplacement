#!/bin/bash

# 执行第一个Python代码
#python save_solution.py

#python experiment_epochs.py


#python ddpg_agent.py

#python td3_agent.py

# 源文件路径
config_file="train_config.yaml"
# 备份文件路径
backup_file="train_config.yaml.bak"
# 备份源文件
cp "$config_file" "$backup_file"

#target_key="epochs"
#new_value="66"
#
#sed -i "s#\(${target_key}:\).*#\1 ${new_value}#g" $config_file


# 要修改的键和值
target_key="epochs"

# 列表中的新值
#new_values=("0"  "20"  "50"  "100")
# 列表中的新值
new_values=()

# 生成96-109范围内的值
for (( i = 96; i <= 109; i++ )); do
  new_values+=("$i")
done
#
# 循环遍历新值列表
for new_value in "${new_values[@]}"; do
  # 使用sed命令修改配置文件中指定键的值

  sed -i "s#\(${target_key}:\).*#\1 ${new_value}#g" $config_file
  cat $config_file
  echo "配置文件已修改为新值: $new_value"
  # 运行测试脚本
  python imitation_learning.py
done




echo "所有的新值已经测试完毕，并且配置文件已经被还原为备份文件"

