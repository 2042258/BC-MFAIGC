import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
# 定义文件夹路径
folder_path = str(BCFLDIF_ROOT.joinpath('data_processing', 'bcfl', 'processing_outcome'))
logs_dir = str(BCFLDIF_ROOT.joinpath('logs', 'flogs'))
nflogs_path = str(BCFLDIF_ROOT.joinpath('logs', 'nflogs'))

# 选择要绘制的 num_malicious 值
selected_malicious_counts = [6]  # 可以根据需要修改
specified_num_malicious_values = [6]  # 你可以在这里指定你需要的值
num_malicious_to_plot = [6]  # 你可以修改这个集合来指定不同的 --num_malicious 值

# 定义一个函数来解析文件内容并提取数据
def parse_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = {}
    for line in lines:
        match = re.match(r'Comm_(\d+)\s+average\s+B\s+value:\s+([\d.]+)', line)
        if match:
            comm_num = int(match.group(1))
            b_value = float(match.group(2))
            data[comm_num] = b_value

    return data

# 定义一个函数来提取文件名中的恶意数量信息
def extract_malicious_count(file_name):
    match = re.search(r'--num_malicious\s+(\d+)', file_name)
    if match:
        return int(match.group(1))
    return None

# 获取文件夹中的所有文件
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 初始化一个字典来存储不同 num_malicious 的数据
malicious_data = {}

# 遍历每个文件并提取数据
for file in files:
    file_path = os.path.join(folder_path, file)
    data = parse_file(file_path)

    # 提取恶意数量
    malicious_count = extract_malicious_count(file)
    if malicious_count is not None:
        if malicious_count not in malicious_data:
            malicious_data[malicious_count] = {}

        # 合并数据
        for comm_num, b_value in data.items():
            if comm_num not in malicious_data[malicious_count]:
                malicious_data[malicious_count][comm_num] = []
            malicious_data[malicious_count][comm_num].append(b_value)

# 计算每个恶意数量的平均值
for malicious_count, data in malicious_data.items():
    for comm_num in data:
        malicious_data[malicious_count][comm_num] = sum(data[comm_num]) / len(data[comm_num])

# 处理 ESM 损失数据
def parse_args_used(args_file_path):
    with open(args_file_path, 'r') as file:
        for line in file:
            if '--num_malicious' in line:
                match = re.search(r'--num_malicious (\d+)', line)
                if match:
                    return int(match.group(1))
    return None

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
    all_trainings_losses = {}

    training_dirs = [d for d in os.listdir(logs_dir) if d.endswith('_ddpm')]
    training_dirs = sorted(training_dirs)

    for training_dir in training_dirs:
        training_dir_path = os.path.join(logs_dir, training_dir)
        if os.path.isdir(training_dir_path):
            args_file_path = os.path.join(training_dir_path, 'args_used.txt')
            num_malicious = parse_args_used(args_file_path)
            if num_malicious is not None:
                round_losses = process_single_training(training_dir_path)
                if round_losses:
                    if num_malicious not in all_trainings_losses:
                        all_trainings_losses[num_malicious] = []
                    all_trainings_losses[num_malicious].append(round_losses)

    return all_trainings_losses

all_trainings_losses = process_all_trainings(logs_dir)

# 处理 nflogs 数据
epoch_losses = {}
num_malicious_losses = {}

for timestamp_folder in os.listdir(nflogs_path):
    timestamp_folder_path = os.path.join(nflogs_path, timestamp_folder)

    if os.path.isdir(timestamp_folder_path):
        args_file_path = os.path.join(timestamp_folder_path, "args_used.txt")
        if os.path.exists(args_file_path):
            with open(args_file_path, "r", encoding='utf-8') as args_file:
                args_content = args_file.read()
                num_malicious_match = re.search(r"--num_malicious (\d+)", args_content)
                if num_malicious_match:
                    num_malicious = int(num_malicious_match.group(1))
                else:
                    continue

        if num_malicious not in specified_num_malicious_values:
            continue

        for i in range(1, 101):
            file_name = f"accuracy_comm_{i}.txt"
            file_path = os.path.join(timestamp_folder_path, file_name)

            if os.path.exists(file_path):
                with open(file_path, "r", encoding='utf-8') as infile:
                    content = infile.read()
                    matches = re.findall(r"Epoch (\d+), Test Loss: ([\d.]+)", content)

                    for epoch, loss in matches:
                        epoch = int(epoch)
                        loss = float(loss)

                        if epoch not in epoch_losses:
                            epoch_losses[epoch] = []

                        epoch_losses[epoch].append(loss)
            else:
                continue

        for epoch, losses in epoch_losses.items():
            if num_malicious not in num_malicious_losses:
                num_malicious_losses[num_malicious] = {}
            if epoch not in num_malicious_losses[num_malicious]:
                num_malicious_losses[num_malicious][epoch] = []
            num_malicious_losses[num_malicious][epoch].extend(losses)

average_losses = {}
for num_malicious, epoch_losses in num_malicious_losses.items():
    average_losses[num_malicious] = {}
    for epoch, losses in epoch_losses.items():
        average_losses[num_malicious][epoch] = sum(losses) / len(losses)

# 初始化图表
plt.figure(figsize=(18, 12))

# 绘制 bcfl 数据
for malicious_count in selected_malicious_counts:
    if malicious_count in malicious_data:
        data = malicious_data[malicious_count]
        comm_nums = sorted(data.keys())
        b_values = [data[comm] for comm in comm_nums]
        plt.plot(comm_nums, b_values, label=f'bcfl-num_malicious {malicious_count}', marker='o')

# 绘制 ESM 损失数据
if all_trainings_losses:
    rounds = range(1, len(next(iter(all_trainings_losses.values()))[0]) + 1)

    for num_malicious in num_malicious_to_plot:
        if num_malicious in all_trainings_losses:
            trainings_to_plot = all_trainings_losses[num_malicious]
            combined_losses = []

            for i in range(len(trainings_to_plot[0])):
                round_losses = [training[i] for training in trainings_to_plot]
                combined_losses.append(np.mean(round_losses))

            plt.plot(rounds, combined_losses, marker='o', linestyle='-', label=f'FL-num_malicious={num_malicious}')

# 绘制 nflogs 数据
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for idx, (num_malicious, epoch_losses) in enumerate(average_losses.items()):
    if num_malicious in specified_num_malicious_values:
        epochs = sorted(epoch_losses.keys())
        losses = [epoch_losses[epoch] for epoch in epochs]
        plt.plot(epochs, losses, marker='o', linestyle='-', color=colors[idx % len(colors)], label=f'trid-num_malicious {num_malicious}')

import os
import re
import matplotlib.pyplot as plt
import numpy as np

# 初始化全局样式设置
plt.style.use('seaborn-darkgrid')  # 使用现代风格预设[1,3](@ref)
plt.rcParams.update({
    'font.family': 'Arial',  # 全局字体设置[3](@ref)
    'font.size': 12,  # 基础字号
    'axes.titlesize': 16,  # 标题字号
    'axes.labelsize': 14,  # 轴标签字号
    'lines.linewidth': 2.5,  # 线宽加粗
    'lines.markersize': 8,  # 标记大小
    'grid.alpha': 0.4,  # 网格透明度
    'figure.dpi': 300,  # 输出分辨率[4](@ref)
})

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# --------------------------
# 全局样式配置（网页6[6](@ref)、网页8[8](@ref)）
# --------------------------
plt.style.use('seaborn-darkgrid')  # 使用专业级网格背景
plt.rcParams.update({
    'font.family': 'Microsoft YaHei',  # 中文字体支持
    'axes.unicode_minus': False,  # 解决负号显示问题
    'axes.prop_cycle': cycler('color', ['#2E86C1', '#C0392B', '#27AE60', '#8E44AD']) +  # 定制颜色循环
                       cycler('linestyle', ['-', '--', '-.', ':']),
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'grid.linestyle': '--',
    'grid.alpha': 0.4
})



def create_combined_plot():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_facecolor('#FDFEFE')  # 坐标系背景色

    # --------------------------
    # BCFL数据绘制
    # --------------------------
    for malicious_count in selected_malicious_counts:
        if malicious_count in malicious_data:
            data = malicious_data[malicious_count]
            comm_nums = sorted(data.keys())
            b_values = [data[comm] for comm in comm_nums]
            ax.plot(comm_nums, b_values,
                    marker='o',
                    markersize=4,
                    linewidth=0.5,
                    alpha=0.8,
                    label=f'BC-MFAIGC')


    if all_trainings_losses:
        rounds = range(1, len(next(iter(all_trainings_losses.values()))[0]) + 1)
        for num_malicious in num_malicious_to_plot:
            if num_malicious in all_trainings_losses:
                trainings_to_plot = all_trainings_losses[num_malicious]
                combined_losses = [np.mean([training[i] for training in trainings_to_plot])
                                   for i in range(len(trainings_to_plot[0]))]
                ax.plot(rounds, combined_losses,
                        marker='s',
                        markersize=4,
                        linestyle='--',
                        linewidth=0.5,
                        alpha=0.8,
                        label=f'Horizontal federated learning')

    # --------------------------
    # Trid数据可视化
    # --------------------------
    colors = ['#3498DB', '#E74C3C']
    for idx, (num_malicious, epoch_losses) in enumerate(average_losses.items()):
        if num_malicious in specified_num_malicious_values:
            epochs = sorted(epoch_losses.keys())
            losses = [epoch_losses[epoch] for epoch in epochs]
            ax.plot(epochs, losses,
                    color=colors[idx % len(colors)],
                    marker='^',
                    markersize=4,
                    linestyle='-.',
                    linewidth=0.5,
                    alpha=0.8,
                    label=f'Traditional train')

    # --------------------------
    # --------------------------
    ax.set_title('Comparative Analysis of 60% Resistance Attacks by Multiple Algorithms', pad=20, fontsize=18, family='Times New Roman')
    ax.set_xlabel('Training Rounds', labelpad=15, family='Times New Roman')
    ax.set_ylabel('Evaluate the index value', labelpad=15, family='Times New Roman')

    #高级图例配置
    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=1.5
    )

    legend.get_frame().set_edgecolor('#2C3E50')

    # 双轴刻度优化
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 动态范围调整
    ax.set_xlim(left=0, auto=True)
    ax.set_ylim(bottom=0, auto=True)

    # 保存与显示
    plt.tight_layout()
    plt.savefig('optimized_plot_6.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# --------------------------
# 执行可视化
# --------------------------
if __name__ == "__main__":
    create_combined_plot()