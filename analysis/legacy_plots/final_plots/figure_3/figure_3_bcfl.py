import os
import matplotlib.pyplot as plt
import numpy as np  
import seaborn as sns
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
# ==================== 全局样式设置 ====================
plt.style.use('seaborn')  # 使用专业风格模板[2](@ref)
plt.rcParams.update({
    'font.family': 'Arial',  # 统一字体类型[2](@ref)
    'axes.labelsize': 12,  # 坐标轴标签字号
    'axes.titlesize': 14,  # 标题字号
    'xtick.labelsize': 10,  # X轴刻度字号
    'ytick.labelsize': 10,  # Y轴刻度字号
    'figure.dpi': 300,  # 输出分辨率[2](@ref)
    'savefig.bbox': 'tight'  # 自动裁剪白边
})

# ==================== 数据准备 ====================
root_dir = str(BCFLDIF_ROOT.joinpath('logs', 'blogs'))

# 使用专业配色方案[2](@ref)
professional_colors = ['#2E86C1', '#138D75', '#B03A2E', '#8E44AD', '#F1C40F']
color_map = {k: professional_colors[(k - 1) % 5] for k in range(1, 11)}  # 循环配色

data_dict = {}

# ==================== 数据加载与处理 ====================
for timestamp_dir in os.listdir(root_dir):
    timestamp_path = os.path.join(root_dir, timestamp_dir)
    if not os.path.isdir(timestamp_path):
        continue

    # 使用生成器表达式优化内存[6](@ref)
    args_file = os.path.join(timestamp_path, 'args_used.txt')
    num_malicious = next((int(line.split()[-1]) for line in open(args_file)
                          if '--num_malicious' in line), None)

    if not num_malicious:
        continue

    comm_b_values = {}
    for comm_num in range(1, 101):
        file_path = os.path.join(timestamp_path,
                                 f'comm_{comm_num}/worker_final_local_accuracies_comm_{comm_num}.txt')

        # 使用矢量化操作优化数据处理[7](@ref)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                b_values = [float(line.split(":")[-1].strip())
                            for line in f if "worker B" in line]
                if b_values:
                    comm_b_values[comm_num] = (
                        np.mean(b_values),  # 平均值
                        np.std(b_values)  # 标准差[2](@ref)
                    )
        except FileNotFoundError:
            continue

    if num_malicious not in data_dict:
        data_dict[num_malicious] = []
    data_dict[num_malicious].append(comm_b_values)

# ==================== 可视化优化 ====================
fig, ax = plt.subplots(figsize=(10, 6))  # 黄金比例尺寸[2](@ref)

# 坐标轴优化
ax.tick_params(direction='in', length=6, width=1)  # 刻度线向内延伸[2](@ref)
ax.grid(True, linestyle='--', alpha=0.6)  # 虚线网格增强可读性

for num_malicious, data_list in data_dict.items():
    color = color_map.get(num_malicious, '#333333')

    # 数据聚合优化
    all_communities = np.array(sorted({k for d in data_list for k in d.keys()}))
    avg_values = []
    std_values = []

    # 使用矩阵运算优化计算[6](@ref)
    for comm in all_communities:
        vals = [d[comm][0] for d in data_list if comm in d]
        avg_values.append(np.mean(vals))
        std_values.append(np.std(vals))

    # 绘制带误差棒的折线图[2](@ref)
    line = ax.errorbar(all_communities, avg_values, yerr=std_values,
                       fmt='o--', lw=1.5, markersize=6, capsize=4,
                       color=color, alpha=0.8,
                       label=f"{num_malicious} Malicious Nodes")



# ==================== 图表修饰 ====================
ax.set_title('Impact of Malicious Nodes on ESM Loss', fontsize=14, pad=20)
ax.set_xlabel('Training Epoch', fontsize=12, labelpad=10)
ax.set_ylabel('ESM Loss', fontsize=12, labelpad=10)

# 图例优化[2](@ref)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,
          title='Malicious Nodes',
          frameon=True,
          shadow=True,
          loc='upper left',
          bbox_to_anchor=(1.02, 1),
          borderaxespad=0.)

# 坐标轴范围优化
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 101, 10))
ax.set_xticklabels([f'Epoch {x}' if x % 20 == 0 else '' for x in np.arange(0, 101, 10)])

# ==================== 输出优化 ====================
plt.tight_layout()
plt.savefig('malicious_nodes_impact.svg', format='svg')  # 矢量图输出[2](@ref)
plt.show()