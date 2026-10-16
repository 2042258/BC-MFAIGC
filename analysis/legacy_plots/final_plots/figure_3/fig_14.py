import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
# ==================== 全局样式设置 ====================
plt.style.use('seaborn')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'xtick.color': 'black',  # 刻度线颜色
    'ytick.color': 'black'  # 刻度线颜色
})

# ==================== 数据准备 ====================
root_dir = str(BCFLDIF_ROOT.joinpath('logs', 'blogs'))
professional_colors = ['#2E86C1', '#138D75', '#B03A2E', '#8E44AD', '#F1C40F']
color_map = {k: professional_colors[(k - 1) % 5] for k in range(1, 11)}
data_dict = {}

# ==================== 数据加载与处理 ====================
for timestamp_dir in os.listdir(root_dir):
    timestamp_path = os.path.join(root_dir, timestamp_dir)
    if not os.path.isdir(timestamp_path):
        continue

    args_file = os.path.join(timestamp_path, 'args_used.txt')
    num_malicious = next((int(line.split()[-1]) for line in open(args_file)
                          if '--num_malicious' in line), None)

    if not num_malicious:
        continue

    comm_b_values = {}
    for comm_num in range(1, 101):
        file_path = os.path.join(timestamp_path,
                                 f'comm_{comm_num}/worker_final_local_accuracies_comm_{comm_num}.txt')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                b_values = [float(line.split(":")[-1].strip())
                            for line in f if "worker B" in line]
                if b_values:
                    comm_b_values[comm_num] = (np.mean(b_values), np.std(b_values))
        except FileNotFoundError:
            continue

    if num_malicious not in data_dict:
        data_dict[num_malicious] = []
    data_dict[num_malicious].append(comm_b_values)

# ==================== 可视化优化 ====================
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('white')

# 创建放大子图（白色背景）
axins = inset_axes(ax, width="35%", height="35%", loc='lower left',
                   borderpad=13, axes_kwargs={'facecolor': 'white'})

# 主图绘制
zoom_data = {}
for num_malicious, data_list in data_dict.items():
    color = color_map.get(num_malicious, '#333333')

    # 数据聚合
    all_communities = np.array(sorted({k for d in data_list for k in d.keys()}))
    avg_values = []
    std_values = []

    for comm in all_communities:
        vals = [d[comm][0] for d in data_list if comm in d]
        avg_values.append(np.mean(vals))
        std_values.append(np.std(vals))

    # 主图误差棒
    main_line = ax.errorbar(all_communities, avg_values, yerr=std_values,
                            fmt='o--', lw=1.5, markersize=6, capsize=4,
                            color=color, alpha=0.8,
                            label=f"{num_malicious} Nodes")

    # 放大子图误差棒
    zoom_line = axins.errorbar(all_communities, avg_values, yerr=std_values,
                               fmt='o--', lw=1, markersize=4, capsize=3,
                               color=color, alpha=0.9)

    zoom_data[num_malicious] = {'x': all_communities, 'y': avg_values, 'std': std_values}

# ==================== 放大区域设置 ====================
zoom_xrange = (5, 15)
y_values = np.concatenate([np.array(d['y'][5:15]) for d in zoom_data.values()])
y_std = np.concatenate([np.array(d['std'][5:15]) for d in zoom_data.values()])
zoom_yrange = (y_values.min() - y_std.max(), y_values.max() + y_std.max())

axins.set_xlim(*zoom_xrange)
axins.set_ylim(*zoom_yrange)
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.grid(True, linestyle='--', alpha=0.7, color='gray')

# 添加放大区域指示框
rect = Rectangle((zoom_xrange[0], zoom_yrange[0]),
                 zoom_xrange[1] - zoom_xrange[0],
                 zoom_yrange[1] - zoom_yrange[0],
                 edgecolor='#34495e',
                 facecolor='none',
                 linestyle='--',
                 linewidth=1.2,
                 zorder=10)
ax.add_patch(rect)

# 添加连接线
con = ConnectionPatch(xyA=(zoom_xrange[1], zoom_yrange[1]), coordsA=ax.transData,
                      xyB=(zoom_xrange[0], zoom_yrange[0]), coordsB=axins.transData,
                      arrowstyle="-", linewidth=1.2, linestyle="--",
                      color="#7f8c8d", zorder=8)
fig.add_artist(con)


# ==================== 坐标轴加深设置 ====================
def deepen_axes_spines(ax):
    """加深坐标轴轴线颜色并调整线宽"""
    # 设置轴线颜色为深黑色（#2D2D2D），线宽1.5
    ax.spines['bottom'].set_color('#2D2D2D')
    ax.spines['left'].set_color('#2D2D2D')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)

    # 隐藏顶部和右侧轴线（可选，根据需求保留）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def deepen_axes_spines_axins(ax):
    """加深坐标轴轴线颜色并调整线宽"""
    # 设置轴线颜色为深黑色（#2D2D2D），线宽1.5
    ax.spines['bottom'].set_color('#2D2D2D')
    ax.spines['left'].set_color('#2D2D2D')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['top'].set_color('#2D2D2D')
    ax.spines['right'].set_color('#2D2D2D')
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    # # 隐藏顶部和右侧轴线（可选，根据需求保留）
    # ax.spines['top'].set_visible(True)
    # ax.spines['right'].set_visible(True)

# 应用加深设置到主图和放大子图
deepen_axes_spines(ax)
deepen_axes_spines_axins(axins)

# ==================== 图表修饰优化 ====================
# ax.set_title('Impact of Malicious Nodes on ESM Loss with Zoomed View',
#              fontsize=14, pad=20, color='black')
ax.set_xlabel('Training Epoch', fontsize=30, labelpad=10, color='black')
ax.set_ylabel('MSE', fontsize=30, labelpad=10, color='black')

ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 101, 10))
ax.set_xticklabels([f'{x}' if x % 10 == 0 else '' for x in np.arange(0, 101, 10)],
                   color='black', fontsize=25)
ax.tick_params(axis='both', which='major', colors='black')
ax.set_yticklabels(ax.get_yticks(), fontsize=25)  # 或者你想要的其他大小
# 主图网格线
ax.grid(True, linestyle='--', alpha=0.7, color='gray')

# 放大图标注
axins.text(0.95, 0.95, 'Epoch 5-15',
           transform=axins.transAxes,
           ha='right', va='top',
           fontsize=25, color='black',
           bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# 图例优化
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,
          frameon=True,
          shadow=True,
          loc='upper left',
          fontsize=20,  # 字体大小
          bbox_to_anchor=(0.8, 1),
          borderaxespad=0.,
          facecolor='white',
          edgecolor='gray',
          labelcolor='black')

# ==================== 输出优化 ====================
plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0, rect=(0, 0, 1, 0.95))
plt.savefig('figure_14.pdf', dpi=300, facecolor='white')
plt.show()