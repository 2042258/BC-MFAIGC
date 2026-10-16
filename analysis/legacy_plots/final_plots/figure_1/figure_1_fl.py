import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
def parse_args_used(args_file_path):
    """
    解析 args_used.txt 文件，提取 --num_malicious 的值
    """
    with open(args_file_path, 'r') as file:
        for line in file:
            if '--num_malicious' in line:
                # 提取 --num_malicious 的值
                match = re.search(r'--num_malicious (\d+)', line)
                if match:
                    return int(match.group(1))
    return None

def calculate_average_loss(file_path):
    # 初始化总损失值和客户端计数器
    total_loss = 0.0
    client_count = 0

    # 打开文件并读取内容
    with open(file_path, 'r') as file:
        for line in file:
            # 使用正则表达式匹配客户端的损失值
            match = re.search(r'client_\d+ [MB]: (\d+\.\d+)', line)
            if match:
                # 提取损失值并累加到总损失值
                loss = float(match.group(1))
                total_loss += loss
                client_count += 1

    # 计算平均损失值
    if client_count > 0:
        average_loss = total_loss / client_count
        return average_loss
    else:
        return None

def process_single_training(base_dir):
    # 初始化每轮的平均损失值列表
    round_losses = []

    # 获取所有 comm_ 目录，并按照数字顺序排序
    comm_dirs = [d for d in os.listdir(base_dir) if d.startswith('comm_')]
    comm_dirs = sorted(comm_dirs, key=lambda x: int(x.split('_')[-1]))  # 按照数字排序

    # 遍历排序后的 comm_ 目录
    for comm_dir in comm_dirs:
        comm_dir_path = os.path.join(base_dir, comm_dir)
        if os.path.isdir(comm_dir_path):
            # 提取轮次编号（例如 comm_1 中的 1）
            comm_round = comm_dir.split('_')[-1]
            # 构建 global_comm_X.txt 文件名
            global_comm_file = f'global_comm_{comm_round}.txt'
            global_comm_path = os.path.join(comm_dir_path, global_comm_file)
            if os.path.isfile(global_comm_path):
                # 计算当前轮次的平均损失值
                average_loss = calculate_average_loss(global_comm_path)
                if average_loss is not None:
                    # 存储当前轮次的平均损失值
                    round_losses.append(average_loss)

    return round_losses

def process_all_trainings(logs_dir):
    # 初始化所有训练的平均损失值字典，按 --num_malicious 分组
    all_trainings_losses = {}

    # 获取所有训练目录
    training_dirs = [d for d in os.listdir(logs_dir) if d.endswith('_ddpm')]
    training_dirs = sorted(training_dirs)  # 按照字母顺序排序

    # 遍历每个训练目录
    for training_dir in training_dirs:
        training_dir_path = os.path.join(logs_dir, training_dir)
        if os.path.isdir(training_dir_path):
            print(f"处理训练目录: {training_dir}")
            # 解析 args_used.txt 文件，获取 --num_malicious 的值
            args_file_path = os.path.join(training_dir_path, 'args_used.txt')
            num_malicious = parse_args_used(args_file_path)
            if num_malicious is not None:
                # 处理单个训练目录
                round_losses = process_single_training(training_dir_path)
                if round_losses:
                    if num_malicious not in all_trainings_losses:
                        all_trainings_losses[num_malicious] = []
                    all_trainings_losses[num_malicious].append(round_losses)

    return all_trainings_losses

def plot_combined_losses(all_trainings_losses, num_malicious_to_plot):
    """
    绘制指定 --num_malicious 值的图表，并标注第100轮的ESM值
    """
    # 获取轮次范围，假设所有训练轮次数相同
    first_trainings = next(iter(all_trainings_losses.values()))
    num_rounds = len(first_trainings[0])
    rounds = range(1, num_rounds + 1)

    plt.figure(figsize=(10, 6))

    # 遍历需要绘制的 --num_malicious 值
    for num_malicious in num_malicious_to_plot:
        if num_malicious in all_trainings_losses:
            trainings_to_plot = all_trainings_losses[num_malicious]

            # 计算每轮的平均损失值
            combined_losses = []
            for i in range(len(trainings_to_plot[0])):
                round_losses = [training[i] for training in trainings_to_plot]
                combined_losses.append(np.mean(round_losses))

            # 绘制当前 --num_malicious 的曲线
            line, = plt.plot(rounds, combined_losses, marker='o', linestyle='-', label=f'ESM (num_malicious={num_malicious})')
            color = line.get_color()

            # 标注并打印第100轮的值
            if len(combined_losses) >= 100:
                round_100 = 100
                idx = round_100 - 1  # 索引为99
                value = combined_losses[idx]
                # 在图表上添加标记和数值
                plt.scatter(round_100, value, color=color, zorder=5)
                plt.text(round_100 + 1, value, f'{value:.2f}', color=color, ha='left', va='center')
                # 打印到控制台
                print(f"num_malicious={num_malicious}, 第100轮ESM值: {value:.2f}")

        else:
            print(f"没有找到 --num_malicious={num_malicious} 的训练数据。")

    # 设置图表标题和标签
    plt.title('epoch_ESM_loss')
    plt.xlabel('Communication Round')
    plt.ylabel('ESM')
    plt.xticks(rounds)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 基础目录路径
logs_dir = str(BCFLDIF_ROOT.joinpath('logs', 'flogs'))

# 处理所有训练目录
all_trainings_losses = process_all_trainings(logs_dir)

if all_trainings_losses:
    # 指定要绘制的 --num_malicious 值集合
    num_malicious_to_plot = [0, 1, 2, 3, 4, 5, 6]  # 你可以修改这个集合来指定不同的 --num_malicious 值
    # 绘制合并后的图表
    plot_combined_losses(all_trainings_losses, num_malicious_to_plot)
else:
    print("没有找到有效的训练数据。")