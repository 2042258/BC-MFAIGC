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
    绘制指定 --num_malicious 值的图表
    """
    rounds = range(1, len(next(iter(all_trainings_losses.values()))[0]) + 1)  # 假设所有训练的轮次数相同

    plt.figure(figsize=(10, 6))

    # 遍历需要绘制的 --num_malicious 值
    for num_malicious in num_malicious_to_plot:
        if num_malicious in all_trainings_losses:
            trainings_to_plot = all_trainings_losses[num_malicious]

            # 初始化每轮的平均损失值列表
            combined_losses = []

            # 遍历每轮
            for i in range(len(trainings_to_plot[0])):
                # 获取当前轮次的损失值
                round_losses = [training[i] for training in trainings_to_plot]
                # 计算当前轮次的平均损失值
                combined_losses.append(np.mean(round_losses))

            # 绘制当前 --num_malicious 的曲线
            plt.plot(rounds, combined_losses, marker='o', linestyle='-', label=f'ESM (num_malicious={num_malicious})')
        else:
            print(f"没有找到 --num_malicious={num_malicious} 的训练数据。")

    # 设置图表标题和标签
    plt.title('epoch_ESM_loss')
    plt.xlabel('commment round')
    plt.ylabel('ESM')
    plt.xticks(rounds)  # 设置 x 轴刻度为轮次编号
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 基础目录路径
logs_dir = str(BCFLDIF_ROOT.joinpath('logs', 'flogs'))

# 处理所有训练目录
all_trainings_losses = process_all_trainings(logs_dir)
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# ================== 全局样式设置 ==================
plt.style.use('seaborn-darkgrid')  # 专业网格样式[2,7](@ref)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',  # 支持中文的字体[2](@ref)
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'grid.alpha': 0.3
})


def plot_combined_losses(all_trainings_losses, num_malicious_to_plot):
    # ================== 画布设置 ==================
    fig = plt.figure(figsize=(14, 8), facecolor='white')  # 16:9专业比例[1](@ref)
    ax = fig.add_subplot(111)

    # ================== 颜色与标记系统 ==================
    colors = plt.cm.viridis(np.linspace(0, 1, len(num_malicious_to_plot)))  # 彩虹色系[7](@ref)
    markers = ['o', 's', 'D', '^', 'v', '<', '>']  # 多样化标记[2](@ref)

    # ================== 数据可视化增强 ==================
    rounds = range(1, len(next(iter(all_trainings_losses.values()))[0]) + 1)

    for idx, num_malicious in enumerate(num_malicious_to_plot):
        if num_malicious in all_trainings_losses:
            trainings_to_plot = all_trainings_losses[num_malicious]
            combined_losses = [np.mean([training[i] for training in trainings_to_plot])
                               for i in range(len(trainings_to_plot[0]))]

            # ========== 主趋势线 ==========
            ax.plot(rounds, combined_losses,
                    marker=markers[idx % 7],
                    markersize=8,
                    markeredgecolor='w',  # 白色标记边框[7](@ref)
                    linestyle='--' if idx % 2 else '-',  # 交替线型[2](@ref)
                    linewidth=2,
                    alpha=0.9,
                    color=colors[idx],
                    label=f'{num_malicious} Malicious Nodes')

            # ========== 置信区间带 ==========
            lower = np.array(combined_losses) * 0.98
            upper = np.array(combined_losses) * 1.02
            ax.fill_between(rounds, lower, upper,
                            color=colors[idx], alpha=0.1)  # 半透明区间带[2](@ref)

    # ================== 坐标轴优化 ==================
    ax.set_xlim(left=0.5, right=len(rounds) + 0.5)  # 扩展坐标范围[1](@ref)
    ax.xaxis.set_major_locator(MultipleLocator(5))  # 每5轮显示主刻度[1](@ref)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylim(bottom=0)  # Y轴从0开始[1](@ref)

    # ================== 辅助元素 ==================
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)  # 关键阈值线[7](@ref)
    ax.grid(True, linestyle='-.', alpha=0.7)  # 虚线网格[2](@ref)

    # ================== 标注系统 ==================
    legend = ax.legend(
        frameon=True,
        loc='upper right',
        ncol=1,
        title='Malicious Nodes',
        title_fontsize=12,
        edgecolor='#E0E0E0',
        bbox_to_anchor=(1.175, 1)  # 右侧外部图例[2](@ref)
    )


    # ================== 标题与标签 ==================
    ax.set_title("Analysis of the Impact of Malicious Nodes on Horizontal Federated Learning Training" ,fontsize=18, pad=20)
    ax.set_xlabel("Training Rounds", fontsize=14, labelpad=10)
    ax.set_ylabel("Average Loss", fontsize=14, labelpad=10)

    # ================== 输出设置 ==================
    plt.tight_layout(pad=3)
    plt.savefig('malicious_nodes_analysis_fl.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')  # 专业保存设置[1](@ref)
    plt.show()


# 其余数据处理函数保持原样...
if all_trainings_losses:
    # 指定要绘制的 --num_malicious 值集合
    num_malicious_to_plot = [0, 1, 2, 3, 4, 5, 6]  # 你可以修改这个集合来指定不同的 --num_malicious 值
    # 绘制合并后的图表
    plot_combined_losses(all_trainings_losses, num_malicious_to_plot)
else:
    print("没有找到有效的训练数据。")