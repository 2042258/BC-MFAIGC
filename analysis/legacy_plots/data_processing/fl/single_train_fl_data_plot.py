import os
import re
import matplotlib.pyplot as plt
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
# 基础目录路径
base_dir = str(BCFLDIF_ROOT.joinpath('logs', 'flogs', '20241221_112020_ddpm'))

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

def process_all_comm_files(base_dir):
    # 初始化总平均损失值和轮次计数器
    total_average_loss = 0.0
    round_count = 0

    # 存储每轮的平均损失值
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
                    # 打印当前轮次的平均损失值
                    print(f"轮次 {comm_dir}/{global_comm_file} 的平均损失值: {average_loss}")
                    total_average_loss += average_loss
                    round_count += 1
                    # 存储当前轮次的平均损失值
                    round_losses.append(average_loss)

    # 计算所有轮次的总平均损失值
    if round_count > 0:
        overall_average_loss = total_average_loss / round_count
        return round_losses, overall_average_loss
    else:
        return None, None

def plot_losses(rounds, losses):
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, losses, marker='o', linestyle='-', color='b', label='ESM')
    plt.title('EMS_comm_loss')
    plt.xlabel('epoch')
    plt.ylabel('ESM')
    plt.xticks(rounds)  # 设置 x 轴刻度为轮次编号
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# 处理所有 comm_ 目录下的 global_comm_X.txt 文件
round_losses, overall_average_loss = process_all_comm_files(base_dir)

if round_losses is not None:
    print(f"所有轮次的总平均损失值: {overall_average_loss}")
    # 绘制图表
    rounds = range(1, len(round_losses) + 1)  # 轮次编号从 1 到 20
    plot_losses(rounds, round_losses)
else:
    print("没有找到有效的损失值。")