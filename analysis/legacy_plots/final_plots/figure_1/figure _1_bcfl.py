import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
# 定义根目录
root_dir = str(BCFLDIF_ROOT.joinpath('logs', 'blogs'))  # 修改为实际的根目录

# 定义颜色映射
color_map = {1: 'blue', 2: 'green', 3: 'red', 4: 'purple', 5: 'orange',
             6: 'cyan', 7: 'magenta', 8: 'brown', 9: 'pink', 10: 'gray'}

# 初始化存储结构
data_dict = {}
final_b_values = {}  # 存储每个num_malicious的第100次社区B值

# 遍历时间戳文件夹
for timestamp_dir in os.listdir(root_dir):
    timestamp_path = os.path.join(root_dir, timestamp_dir)
    if not os.path.isdir(timestamp_path):
        continue

    # 提取--num_malicious值
    num_malicious = None
    args_file = os.path.join(timestamp_path, 'args_used.txt')
    if os.path.exists(args_file):
        with open(args_file, 'r') as file:
            for line in file:
                if '--num_malicious' in line:
                    num_malicious = int(line.split('--num_malicious ')[1].split()[0])
                    break

    if num_malicious is None:
        continue

    # 处理每个社区
    comm_b_values = {}
    for comm_num in range(1, 101):
        comm_dir = os.path.join(timestamp_path, f'comm_{comm_num}')
        file_path = os.path.join(comm_dir, f'worker_final_local_accuracies_comm_{comm_num}.txt')

        if not os.path.exists(file_path):
            continue

        # 提取B值
        b_values = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if "worker B" in line:
                    b_values.append(float(line.split(":")[-1].strip()))

        if b_values:
            comm_b_values[comm_num] = np.mean(b_values)

    # 存储数据
    if num_malicious is not None:
        data_dict.setdefault(num_malicious, []).append(comm_b_values)

        # 记录第100次社区的值
        if 100 in comm_b_values:
            final_b_values.setdefault(num_malicious, []).append(comm_b_values[100])

# 绘制折线图（原有绘图逻辑）
plt.figure(figsize=(12, 8))
for num_mal, color in color_map.items():
    if num_mal not in data_dict:
        continue
    # 合并数据并计算平均值
    all_comm = []
    all_b = []
    for comm_data in data_dict[num_mal]:
        for comm, b in comm_data.items():
            all_comm.append(comm)
            all_b.append(b)
    # 排序及处理
    all_comm, all_b = np.array(all_comm), np.array(all_b)
    sorted_idx = np.argsort(all_comm)
    unique_comm, indices = np.unique(all_comm[sorted_idx], return_index=True)
    avg_b = [np.mean(all_b[sorted_idx][indices[i]:indices[i + 1]]) if i < len(indices) - 1
             else np.mean(all_b[sorted_idx][indices[i]:]) for i in range(len(indices))]
    # 绘图
    plt.plot(unique_comm, avg_b, marker='o', linestyle='-', color=color, label=f"{num_mal} Nodes")

plt.title('The Impact of Different Noise Nodes', fontsize=16)
plt.xlabel('Community Epoch', fontsize=14)
plt.ylabel('ESM Loss', fontsize=14)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.grid(True)
plt.xticks(np.arange(0, 101, 5))
plt.show()

# 输出第100次社区的B值统计信息
print("第100次社区B值统计：")
for num_mal in sorted(final_b_values.keys()):
    b_values = final_b_values[num_mal]
    if not b_values:
        continue
    print(f"--num_malicious {num_mal}:")
    print(f"  样本数: {len(b_values)}")
    print(f"  平均值: {np.mean(b_values):.4f} ± {np.std(b_values):.4f}")
    print(f"  数值列表: {[round(x, 4) for x in b_values]}\n")