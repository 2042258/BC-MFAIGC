import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
# 第一幅图的数据提取和绘制
def plot_first_graph():
    root_dir = str(BCFLDIF_ROOT.joinpath('logs', 'blogs'))  # 修改为实际的根目录
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
    data_dict = {}

    for timestamp_dir in os.listdir(root_dir):
        timestamp_path = os.path.join(root_dir, timestamp_dir)
        if not os.path.isdir(timestamp_path):
            continue

        args_file = os.path.join(timestamp_path, 'args_used.txt')
        num_malicious = None
        if os.path.exists(args_file):
            with open(args_file, 'r') as file:
                for line in file:
                    if '--num_malicious' in line:
                        num_malicious = int(line.split('--num_malicious ')[1].split()[0])
                        break

        if num_malicious is None:
            print(f"Warning: --num_malicious not found in {args_file}. Skipping {timestamp_dir}.")
            continue

        comm_b_values = {}
        for comm_num in range(1, 21):
            comm_dir = os.path.join(timestamp_path, f'comm_{comm_num}')
            file_path = os.path.join(comm_dir, f'worker_final_local_accuracies_comm_{comm_num}.txt')

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            b_values = []
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if "worker B" in line:
                        value = float(line.split(":")[-1].strip())
                        b_values.append(value)

            if b_values:
                average_b_value = sum(b_values) / len(b_values)
                comm_b_values[comm_num] = average_b_value
            else:
                print(f"No B values found in Comm_{comm_num}")

        if num_malicious is not None:
            if num_malicious not in data_dict:
                data_dict[num_malicious] = []
            data_dict[num_malicious].append(comm_b_values)

    if 4 in data_dict:
        num_malicious = 2
        data_list = data_dict[num_malicious]
        color = color_map.get(num_malicious, 'black')

        all_communities = []
        all_b_values = []

        for comm_b_values in data_list:
            for comm_num, b_value in comm_b_values.items():
                all_communities.append(comm_num)
                all_b_values.append(b_value)

        all_communities = np.array(all_communities)
        all_b_values = np.array(all_b_values)

        sorted_indices = np.argsort(all_communities)
        sorted_communities = all_communities[sorted_indices]
        sorted_b_values = all_b_values[sorted_indices]

        unique_communities, indices = np.unique(sorted_communities, return_index=True)
        average_b_values = np.array([np.mean(sorted_b_values[indices[i]:indices[i+1]]) if i < len(indices)-1 else np.mean(sorted_b_values[indices[i]:]) for i in range(len(indices))])

        return unique_communities, average_b_values, color, "bcfl_loss"

# 第二幅图的数据提取和绘制
def plot_second_graph():
    file_path = str(BCFLDIF_ROOT.joinpath('data_processing', 'tradi', 'average_loss', 'average_losses.txt'))
    epochs = []
    losses = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(", ")
            if len(parts) == 2:
                epoch = int(parts[0].split(":")[0].split(" ")[1])
                loss = float(parts[1].split(":")[1])
                epochs.append(epoch)
                losses.append(loss)

    return epochs, losses, 'b', "tradi_loss"

# 第三幅图的数据提取和绘制
def plot_third_graph():
    def calculate_average_loss(file_path):
        total_loss = 0.0
        client_count = 0

        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(r'client_\d+ [MB]: (\d+\.\d+)', line)
                if match:
                    loss = float(match.group(1))
                    total_loss += loss
                    client_count += 1

        if client_count > 0:
            average_loss = total_loss / client_count
            return average_loss
        else:
            return None

    def process_single_training(base_dir):
        round_losses = []
        comm_dirs = [d for d in os.listdir(base_dir) if d.startswith('comm_')]
        comm_dirs = sorted(comm_dirs, key=lambda x: int(x.split('_')[-1]))

        for comm_dir in comm_dirs:
            comm_dir_path = os.path.join(base_dir, comm_dir)
            if os.path.isdir(comm_dir_path):
                comm_round = comm_dir.split('_')[-1]
                global_comm_file = f'global_comm_{comm_round}.txt'
                global_comm_path = os.path.join(comm_dir_path, global_comm_file)
                if os.path.isfile(global_comm_path):
                    average_loss = calculate_average_loss(global_comm_path)
                    if average_loss is not None:
                        round_losses.append(average_loss)

        return round_losses

    def process_all_trainings(logs_dir):
        all_trainings_losses = []
        training_dirs = [d for d in os.listdir(logs_dir) if d.endswith('_improved_diffusion_model')]
        training_dirs = sorted(training_dirs)

        for training_dir in training_dirs:
            training_dir_path = os.path.join(logs_dir, training_dir)
            if os.path.isdir(training_dir_path):
                print(f"处理训练目录: {training_dir}")
                round_losses = process_single_training(training_dir_path)
                if round_losses:
                    all_trainings_losses.append(round_losses)

        return all_trainings_losses

    logs_dir = str(BCFLDIF_ROOT.joinpath('logs', 'flogs'))
    all_trainings_losses = process_all_trainings(logs_dir)

    if all_trainings_losses:
        combined_losses = []
        for i in range(len(all_trainings_losses[0])):
            round_losses = [training[i] for training in all_trainings_losses]
            combined_losses.append(np.mean(round_losses))

        rounds = range(1, len(combined_losses) + 1)
        return rounds, combined_losses, 'r', "fl_loss"

# 合并三幅图的数据并绘制
def plot_combined_graphs():
    plt.figure(figsize=(14, 10))

    # 第一幅图
    x1, y1, color1, label1 = plot_first_graph()
    plt.plot(x1, y1, marker='o', linestyle='-', color=color1, label=label1)

    # 第二幅图
    x2, y2, color2, label2 = plot_second_graph()
    plt.plot(x2, y2, marker='o', linestyle='-', color=color2, label=label2)

    # 第三幅图
    x3, y3, color3, label3 = plot_third_graph()
    plt.plot(x3, y3, marker='o', linestyle='-', color=color3, label=label3)

    # 设置图表标题和标签
    plt.title('Combined Graphs', fontsize=16)
    plt.xlabel('Epoch/Community Round', fontsize=14)
    plt.ylabel('Loss/ESM', fontsize=14)

    # 设置图例
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))

    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.6)

    # 显示图表
    plt.tight_layout()
    plt.show()

# 调用合并绘图函数
plot_combined_graphs()