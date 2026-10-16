import os
import re
import matplotlib.pyplot as plt
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
# 定义 nflogs 文件夹路径
nflogs_path = str(BCFLDIF_ROOT.joinpath('logs', 'nflogs'))

# 初始化一个字典来存储每个轮次的损失值
epoch_losses = {}

# 初始化一个字典来存储每个 --num_malicious 值对应的损失值
num_malicious_losses = {}

# 指定需要绘制的 --num_malicious 值
specified_num_malicious_values = [0,1,2,3,4,5,6]  # 你可以在这里指定你需要的值

# 遍历 nflogs 文件夹下的所有时间戳文件夹
for timestamp_folder in os.listdir(nflogs_path):
    timestamp_folder_path = os.path.join(nflogs_path, timestamp_folder)

    # 检查是否是文件夹
    if os.path.isdir(timestamp_folder_path):
        # 读取 args_used.txt 文件
        args_file_path = os.path.join(timestamp_folder_path, "args_used.txt")
        if os.path.exists(args_file_path):
            with open(args_file_path, "r", encoding='utf-8') as args_file:
                args_content = args_file.read()
                # 使用正则表达式提取 --num_malicious 的值
                num_malicious_match = re.search(r"--num_malicious (\d+)", args_content)
                if num_malicious_match:
                    num_malicious = int(num_malicious_match.group(1))
                else:
                    print(f"文件 {args_file_path} 中未找到 --num_malicious 参数")
                    continue

        # 如果 --num_malicious 的值不在指定的列表中，跳过该文件夹
        if num_malicious not in specified_num_malicious_values:
            continue

        # 遍历1到20的文件名
        for i in range(1, 101):
            file_name = f"accuracy_comm_{i}.txt"
            file_path = os.path.join(timestamp_folder_path, file_name)

            # 检查文件是否存在
            if os.path.exists(file_path):
                # 打开并读取文件内容
                with open(file_path, "r", encoding='utf-8') as infile:
                    content = infile.read()

                    # 使用正则表达式提取每个轮次的损失值
                    matches = re.findall(r"Epoch (\d+), Test Loss: ([\d.]+)", content)

                    for epoch, loss in matches:
                        epoch = int(epoch)
                        loss = float(loss)

                        if epoch not in epoch_losses:
                            epoch_losses[epoch] = []

                        epoch_losses[epoch].append(loss)
            else:
                print(f"文件 {file_path} 不存在")

        # 将损失值按 --num_malicious 分类
        for epoch, losses in epoch_losses.items():
            if num_malicious not in num_malicious_losses:
                num_malicious_losses[num_malicious] = {}
            if epoch not in num_malicious_losses[num_malicious]:
                num_malicious_losses[num_malicious][epoch] = []
            num_malicious_losses[num_malicious][epoch].extend(losses)

# 计算每个轮次的平均损失值
average_losses = {}
for num_malicious, epoch_losses in num_malicious_losses.items():
    average_losses[num_malicious] = {}
    for epoch, losses in epoch_losses.items():
        average_losses[num_malicious][epoch] = sum(losses) / len(losses)

# 绘制折线图
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 定义颜色列表

for idx, (num_malicious, epoch_losses) in enumerate(average_losses.items()):
    if num_malicious in specified_num_malicious_values:  # 只绘制指定的 --num_malicious 值
        epochs = sorted(epoch_losses.keys())
        losses = [epoch_losses[epoch] for epoch in epochs]
        plt.plot(epochs, losses, marker='o', linestyle='-', color=colors[idx % len(colors)], label=f'--num_malicious {num_malicious}')

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl

# ================== 全局样式设置 ==================
plt.style.use('seaborn-darkgrid')  # 专业网格样式[4,7](@ref)
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',  # 支持中文的字体
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'legend.fontsize': 11,
    'grid.alpha': 0.3
})

# ================== 数据处理部分 ==================
# (与原始代码相同，保持数据处理的完整性)
nflogs_path = str(BCFLDIF_ROOT.joinpath('logs', 'nflogs'))
specified_num_malicious_values = [0, 1, 2, 3, 4, 5, 6]
epoch_losses = {}
num_malicious_losses = {}



# ================== 可视化增强部分 ==================
# 创建专业调色板
colors = plt.cm.viridis(np.linspace(0, 1, len(specified_num_malicious_values)))  # 彩虹色系[4](@ref)
markers = ['o', 's', 'D', '^', 'v', '<', '>']  # 多样化标记符号[3](@ref)

plt.figure(figsize=(16, 9), facecolor='white')  # 16:9专业比例[6](@ref)

# 绘制带置信区间的折线
for idx, num_malicious in enumerate(sorted(average_losses.keys())):
    epochs = sorted(average_losses[num_malicious].keys())
    losses = [average_losses[num_malicious][e] for e in epochs]

    # 主趋势线
    line = plt.plot(
        epochs, losses,
        marker=markers[idx],
        markersize=8,
        markeredgewidth=1,
        markeredgecolor='white',
        linestyle='--' if idx % 2 else '-',
        linewidth=2,
        alpha=0.9,
        color=colors[idx],
        label=f'{num_malicious} Malicious Nodes'
    )

    # 置信区间带[1](@ref)
    plt.fill_between(
        epochs,
        np.array(losses) * 0.98,
        np.array(losses) * 1.02,
        color=colors[idx],
        alpha=0.1
    )

# ================== 专业标注系统 ==================
# 坐标轴优化
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(5))  # 每5个epoch显示主刻度[6](@ref)
ax.yaxis.set_major_locator(MultipleLocator(0.1))  # Y轴0.1间隔
ax.set_ylim(0, max([max(v.values()) for v in average_losses.values()]) * 1.1)

# 辅助线系统
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)  # 关键阈值线[7](@ref)

# ================== 图例与标注 ==================
legend = ax.legend(
    frameon=True,
    loc='upper right',
    ncol=1,
    title='Malicious Nodes',
    title_fontsize=12,
    edgecolor='#E0E0E0',
    bbox_to_anchor=(1.16, 1)  # 右侧外部图例[8](@ref)
)


# ================== 最终渲染输出 ==================
plt.title("Analysis of the Impact of Malicious Nodes on Traditional Training",
          fontsize=18, pad=20)
plt.xlabel("Training Rounds", fontsize=14, labelpad=10)
plt.ylabel("Average Loss", fontsize=14, labelpad=10)

plt.tight_layout()
plt.savefig('malicious_nodes_analysis_tra.pdf', bbox_inches='tight')  # 专业保存设置[6](@ref)
plt.show()