import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl

# ================== 全局样式设置 ==================
# 设置全局字体为Times New Roman
mpl.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.labelsize': 30,
    'axes.titlesize': 30,
    'xtick.labelsize': 38,
    'ytick.labelsize': 38,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'legend.fontsize': 20,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# ================== 数据处理部分 ==================
nflogs_path = r"./logs/nflogs"
specified_num_malicious_values = [0, 1, 2, 3, 4, 5, 6]
num_malicious_losses = {}  # 仅保留此全局字典

# 遍历时间戳文件夹
for timestamp_folder in os.listdir(nflogs_path):
    timestamp_folder_path = os.path.join(nflogs_path, timestamp_folder)
    if not os.path.isdir(timestamp_folder_path):
        continue

    # 解析恶意节点数量
    num_malicious = None
    args_file_path = os.path.join(timestamp_folder_path, "args_used.txt")
    if os.path.exists(args_file_path):
        with open(args_file_path, "r", encoding='utf-8') as args_file:
            args_content = args_file.read()
            num_malicious_match = re.search(r"--num_malicious (\d+)", args_content)
            if num_malicious_match:
                num_malicious = int(num_malicious_match.group(1))

    # 过滤不符合的恶意节点数量
    if num_malicious is None or num_malicious not in specified_num_malicious_values:
        continue

    # 为每个文件夹创建独立的损失字典
    folder_epoch_losses = {}

    # 处理每个通信轮次文件
    for comm_round in range(1, 101):
        file_name = f"accuracy_comm_{comm_round}.txt"
        file_path = os.path.join(timestamp_folder_path, file_name)
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r", encoding='utf-8') as infile:
            content = infile.read()
            matches = re.findall(r"Epoch (\d+), Test Loss: ([\d.]+)", content)
            for epoch, loss in matches:
                epoch = int(epoch)
                loss = float(loss)
                if epoch not in folder_epoch_losses:
                    folder_epoch_losses[epoch] = []
                folder_epoch_losses[epoch].append(loss)

    # 将文件夹数据聚合到全局结构
    if num_malicious not in num_malicious_losses:
        num_malicious_losses[num_malicious] = {}

    for epoch, losses in folder_epoch_losses.items():
        if epoch not in num_malicious_losses[num_malicious]:
            num_malicious_losses[num_malicious][epoch] = []
        num_malicious_losses[num_malicious][epoch].extend(losses)

# 计算平均损失
average_losses = {}
for num_malicious, epoch_data in num_malicious_losses.items():
    average_losses[num_malicious] = {}
    for epoch, losses in epoch_data.items():
        average_losses[num_malicious][epoch] = np.mean(losses)

# ================== 可视化增强部分 ==================
# 创建专业调色板
colors = plt.cm.tab10(np.linspace(0, 1, len(specified_num_malicious_values)))
markers = ['o', 's', 'D', '^', 'v', '<', '>']  # 标记符号

plt.figure(figsize=(18, 10), facecolor='white')  # 显式设置画布背景为白色

# 绘制带置信区间的折线
for idx, num_malicious in enumerate(sorted(average_losses.keys())):
    epochs = sorted(average_losses[num_malicious].keys())
    losses = [average_losses[num_malicious][e] for e in epochs]

    # 主趋势线（缩小标记点大小）
    plt.plot(
        epochs, losses,
        marker=markers[idx],
        markersize=6,  # 坐标点（标记）缩小
        markeredgewidth=1,
        markeredgecolor='white',
        linestyle='--' if idx % 2 else '-',
        linewidth=2,
        alpha=0.9,
        color=colors[idx],
        label=f'{num_malicious} Nodes'
    )

    # 添加置信区间带
    plt.fill_between(
        epochs,
        np.array(losses) * 0.98,
        np.array(losses) * 1.02,
        color=colors[idx],
        alpha=0.1
    )

# ================== 专业标注系统 ==================
ax = plt.gca()

# 坐标轴刻度优化（Y轴起始点设为-0.2）
y_max = max([max(v.values()) for v in average_losses.values()]) * 1.1
ax.set_ylim(bottom=-0.03, top=y_max)  # 纵坐标起始点-0.2

# 横坐标起始点设为0
ax.set_xlim(left=0)

# 设置网格线样式
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')

# 去除上方和右侧的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 刻度间隔优化
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# ================== 图例与标注 ==================
legend = ax.legend(
    frameon=True,
    loc='upper right',
    ncol=1,
    fontsize=30,  # 字体大小
    edgecolor='#E0E0E0',
    bbox_to_anchor=(0.94, 0.63)
)

# ================== 最终渲染输出 ==================
# plt.title("Analysis of the Impact of Malicious Nodes on Traditional Training",
#           fontsize=30, pad=20, fontname='Times New Roman')
plt.xlabel("Training Rounds", fontsize=48, labelpad=10, fontname='Times New Roman')
plt.ylabel("MSE", fontsize=48, labelpad=10, fontname='Times New Roman')

plt.tight_layout()
plt.savefig('figure_12_modified.pdf', bbox_inches='tight')
plt.show()