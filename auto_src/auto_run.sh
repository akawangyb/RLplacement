#!/bin/bash

## 执行第一个Python代码
#python save_solution.py
#
#python experiment_epochs.py
#
## 执行第二个Python代码
#python ddpg_agent.py
#
## 执行第三个Python代码
#python td3_agent.py
#
## 执行第四个Python代码
#python imitation_learning.py
#
#python imitation_learning_td3.py



# 源文件路径
config_file="train_config.yaml"
# 备份文件路径
backup_file="train_config.yaml.bak"

# 备份源文件
cp "$config_file" "$backup_file"

# 要修改的键和值
target_key="epochs"
#new_value="new_value"

# 列表中的新值
# 80,90,100
new_values=("99" )


# 循环遍历新值列表
for new_value in "${new_values[@]}"; do
  # 使用sed命令修改配置文件中指定键的值
#  sed -i "s/target_key:.*/target_key: $new_value/" "$config_file"
#  sed -i "s/$target_key:.*/$target_key: $new_value/" "$config_file"

  sed -i "s#\(${target_key}:\).*#\1 ${new_value}#g" $config_file
  cat $config_file

  echo "配置文件已修改为新值: $new_value"

  # 运行测试脚本
  python imitation_learning_td3.py
  python imitation_learning.py
#  bash test_script.sh

done

echo "所有的新值已经测试完毕，并且配置文件已经被还原为备份文件"

