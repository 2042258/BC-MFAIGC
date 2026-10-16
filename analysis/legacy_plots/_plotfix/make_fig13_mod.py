import os
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# ================== 全局样式设置 ==================
plt.rcParams.update({
    'font.family': 'Times New Roman',  # 使用Times New Roman字体
    'axes.labelsize': 20,  # 坐标轴标签字体尺寸
    'axes.titlesize': 20,  # 标题字体尺寸
    'xtick.labelsize': 25,  # x轴刻度字体尺寸
    'ytick.labelsize': 25,  # y轴刻度字体尺寸
    'figure.dpi': 300,  # 图形分辨率
    'savefig.dpi': 300,  # 保存图形分辨率
    'legend.fontsize': 20,
    'axes.edgecolor': 'black',  # 坐标轴颜色
    'axes.linewidth': 0.8,  # 坐标轴线宽
    'grid.linewidth': 0.4,  # 网格线宽
    'grid.alpha': 0.5,  # 网格透明度
    'legend.fontsize': 40,  # 图例字体尺寸
    'figure.facecolor': 'white',  # 图形背景色
    'axes.facecolor': 'white',  # 坐标区域背景色
})


def parse_args_used(args_file_path):
    """
    解析 args_used.txt 文件，提取 --num_malicious 的值
    """
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
        return total_loss / client_count
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


def plot_combined_losses(all_trainings_losses, num_malicious_to_plot):
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    ax = fig.add_subplot(111)

    # 移除非必要的边框
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(num_malicious_to_plot)))
    markers = ['o', 's', 'D', '^', 'v', '<', '>']
    rounds = range(1, len(next(iter(all_trainings_losses.values()))[0]) + 1)

    for idx, num_malicious in enumerate(num_malicious_to_plot):
        if num_malicious in all_trainings_losses:
            trainings_to_plot = all_trainings_losses[num_malicious]
            combined_losses = [np.mean([training[i] for training in trainings_to_plot])
                               for i in range(len(trainings_to_plot[0]))]

            # 使用更小的标记尺寸
            ax.plot(rounds, combined_losses,
                    marker=markers[idx % 7],
                    markersize=5,  # 减小坐标点尺寸
                    markeredgecolor='w',
                    markeredgewidth=0.5,
                    linestyle='--' if idx % 2 else '-',
                    linewidth=1.5,
                    alpha=0.9,
                    color=colors[idx],
                    label=f'{num_malicious} Nodes')

    # 增强网格显示
    ax.grid(True, linestyle='-', alpha=0.4)


    # 优化坐标轴
    ax.set_xlim(left=0.5, right=len(rounds) + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylim(bottom=0)

    # 标题和标签
    # ax.set_title("Impact of Malicious Nodes on Training Loss", fontsize=30)
    ax.set_xlabel("Training Rounds", fontsize=30)
    ax.set_ylabel("MSE", fontsize=30)

    # 图例位置调整
    ax.legend(frameon=True, loc='center right', ncol=1, fontsize=20,bbox_to_anchor=(1, 0.34))

    plt.tight_layout()
    plt.savefig('figure_13_modified.pdf',
                dpi=300,
                bbox_inches='tight',
                facecolor='white'
                )
    plt.show()


# 主程序执行
if __name__ == "__main__":
    logs_dir = r'./logs/flogs'
    all_trainings_losses = process_all_trainings(logs_dir)

    if all_trainings_losses:
        num_malicious_to_plot = [0, 1, 2, 3, 4, 5, 6]
        plot_combined_losses(all_trainings_losses, num_malicious_to_plot,)
    else:
        print("没有找到有效的训练数据。")
