import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
# 文件夹路径
folder_path = str(BCFLDIF_ROOT.joinpath('data_processing', 'bcfl', 'processing_outcome'))

# 初始化字典来存储不同 `--num_malicious` 值对应的社区编号和 B 值
data_dict = {}

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

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        # 提取 `--num_malicious` 后的数字
        num_malicious = None
        if "--num_malicious" in file_name:
            num_malicious = int(file_name.split("--num_malicious ")[1].split(".")[0])

        # 初始化列表来存储社区编号和 B 值
        communities = []
        b_values = []

        # 打开并读取文件
        with open(os.path.join(folder_path, file_name), 'r') as file:
            for line in file:
                # 提取社区编号和 B 值
                if line.startswith("Comm_"):
                    parts = line.split()
                    community = int(parts[0].split('_')[1])  # 提取社区编号
                    b_value = float(parts[-1])  # 提取 B 值
                    communities.append(community)
                    b_values.append(b_value)

        # 将数据存储到字典中
        if num_malicious is not None:
            if num_malicious not in data_dict:
                data_dict[num_malicious] = []
            data_dict[num_malicious].append((communities, b_values))  # 不再存储文件名

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
    for communities, b_values in data_list:
        all_communities.extend(communities)
        all_b_values.extend(b_values)

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
plt.xticks([0, 5, 10, 15, 20])

# 显示图表
plt.show()