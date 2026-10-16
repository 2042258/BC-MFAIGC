import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
# bcdl完整版的画图
# 从原始日志文件中提取数据

# 定义根目录
root_dir = str(BCFLDIF_ROOT.joinpath('logs', 'blogs'))  # 修改为实际的根目录

# 定义颜色映射（可以根据需要扩展颜色）
color_map = {
    1: 'blue',
    2: 'green',
    3: 'red',
    4: 'purple',
    5: 'orange',
    6: 'cyan',
    7: 'magenta',
    8: 'brown',
    9: 'pink',
    10: 'gray'
}

# 初始化字典来存储不同 `--num_malicious` 值对应的社区编号和 B 值
data_dict = {}

# 遍历blogs目录下所有时间戳文件夹
for timestamp_dir in os.listdir(root_dir):
    timestamp_path = os.path.join(root_dir, timestamp_dir)

    # 检查是否是时间戳文件夹
    if not os.path.isdir(timestamp_path):
        continue

    # 从args_used.txt中提取--num_malicious的值
    args_file = os.path.join(timestamp_path, 'args_used.txt')
    num_malicious = None
    if os.path.exists(args_file):
        with open(args_file, 'r') as file:
            for line in file:
                if '--num_malicious' in line:
                    # 提取 --num_malicious 后的数字
                    num_malicious = int(line.split('--num_malicious ')[1].split()[0])
                    break

    # 检查是否成功提取--num_malicious
    if num_malicious is None:
        print(f"Warning: --num_malicious not found in {args_file}. Skipping {timestamp_dir}.")
        continue

    # 初始化一个字典来存储每个comm的B:后的数值
    comm_b_values = {}

    # 遍历comm_1到comm_20目录
    for comm_num in range(1, 101):  # 修改为21以包含comm_20
        comm_dir = os.path.join(timestamp_path, f'comm_{comm_num}')
        file_path = os.path.join(comm_dir, f'worker_final_local_accuracies_comm_{comm_num}.txt')  # 添加.txt后缀

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")  # 打印未找到的文件
            continue

        # 初始化当前comm的B:后的数值列表
        b_values = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if "worker B" in line:
                    value = float(line.split(":")[-1].strip())
                    b_values.append(value)

        # 计算当前comm的B:后数值的平均值
        if b_values:
            average_b_value = sum(b_values) / len(b_values)
            comm_b_values[comm_num] = average_b_value
        else:
            print(f"No B values found in Comm_{comm_num}")

    # 将数据存储到字典中
    if num_malicious is not None:
        if num_malicious not in data_dict:
            data_dict[num_malicious] = []
        data_dict[num_malicious].append(comm_b_values)

# 绘制折线图
plt.figure(figsize=(12, 8))

# 遍历字典中的数据
for num_malicious, data_list in data_dict.items():
    # 获取对应的颜色
    color = color_map.get(num_malicious, 'black')  # 如果 `--num_malicious` 值不在颜色映射中，使用黑色

    # 初始化列表来存储所有社区编号和 B 值
    all_communities = []
    all_b_values = []

    # 合并所有文件的数据
    for comm_b_values in data_list:
        for comm_num, b_value in comm_b_values.items():
            all_communities.append(comm_num)
            all_b_values.append(b_value)

    # 将社区编号和 B 值转换为 NumPy 数组
    all_communities = np.array(all_communities)
    all_b_values = np.array(all_b_values)

    # 按社区编号排序
    sorted_indices = np.argsort(all_communities)
    sorted_communities = all_communities[sorted_indices]
    sorted_b_values = all_b_values[sorted_indices]

    # 计算每个社区编号的平均 B 值
    unique_communities, indices = np.unique(sorted_communities, return_index=True)
    average_b_values = np.array([np.mean(sorted_b_values[indices[i]:indices[i+1]]) if i < len(indices)-1 else np.mean(sorted_b_values[indices[i]:]) for i in range(len(indices))])

    # 绘制平均值的线
    plt.plot(unique_communities, average_b_values, marker='o', linestyle='-', color=color,
             label=f"{num_malicious} Nodes")  # 只显示 num_malicious

# 设置图表标题和标签
plt.title('The Impact of Different Noise Nodes', fontsize=16)
plt.xlabel('Community Epoch', fontsize=14)
plt.ylabel('ESM Loss', fontsize=14)

# 将图例放到图片外部
plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

plt.grid(True)

# 设置横坐标的刻度为 0, 5, 10, 15, 20
ticks = np.arange(0, 101, 5)
plt.xticks(ticks, ticks)
# plt.xticks([0, 5, 10, 15, 20])

# 显示图表
plt.show()