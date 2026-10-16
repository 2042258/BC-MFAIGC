import matplotlib.pyplot as plt
import numpy as np

values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# 转换为带百分号的字符串数组
num_malicious = [f"{int(v*100)}%" for v in values]  # 输出 ['0%', '10%', '20%', ...]
loss_values = [0.0002, 0.8842, 1.0073, 1.1504, 0.9166, 1.0448, 1.0847]  # 原始损失值（保留四位）
esm_values = [0.06, 1.34, 1.17, 1.42, 1.25, 1.32, 1.41]                # ESM值（保留两位）
avg_values = [0.0002]*7                                                 # 新增平均值（保留四位）

import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 22  # 设置全局字体大小

# 创建图形
plt.figure(figsize=(14, 8), dpi=300)
plt.style.use('seaborn-whitegrid')
ax = plt.gca()

# 绘制三条数据线
plt.plot(num_malicious, loss_values,
         linestyle='--', color='#2C6E91', marker='o', markersize=8,
         linewidth=2, label='TRA')

plt.plot(num_malicious, esm_values,
         linestyle='--', color='#D1495B', marker='s', markersize=8,
         linewidth=2, label='FL')

plt.plot(num_malicious, avg_values,
         linestyle='--', color='#2E8B57', marker='D', markersize=8,
         linewidth=2, label='BC-MFAIGC')

# 设置刻度和轴标签
plt.xticks(num_malicious, fontsize=20)
plt.ylim(-0.2, 2.1)
plt.yticks(fontsize=20)

# 设置轴标签字体（Times New Roman，加粗）
plt.xlabel('The proportion of malicious nodes', family='Times New Roman',fontsize=30)
plt.ylabel('MSE',family='Times New Roman', fontsize=30)

# 设置坐标轴刻度字体
for label in ax.get_xticklabels():
    label.set_fontproperties(mpl.font_manager.FontProperties(family='Times New Roman', size=30))

for label in ax.get_yticklabels():
    label.set_fontproperties(mpl.font_manager.FontProperties(family='Times New Roman', size=30))

# 设置图例（Times New Roman）
legend = plt.legend(loc='upper left', frameon=True)
plt.setp(legend.get_texts(), fontfamily='Times New Roman', fontsize=30)

# 美化图形
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存并显示
plt.savefig('figure_10.pdf', bbox_inches='tight')
plt.show()
# plt.figure(figsize=(14, 8), dpi=300)
# plt.style.use('seaborn-whitegrid')
# ax = plt.gca()
#
# plt.plot(num_malicious, loss_values,
#          linestyle='--', color='#2C6E91', marker='o', markersize=8,
#          linewidth=2, label='TRA')
#
# # 2. ESM值（红色虚线，标记方形）
# plt.plot(num_malicious, esm_values,
#          linestyle='--', color='#D1495B', marker='s', markersize=8,
#          linewidth=2, label='FL')
#
# # 3. 平均值（绿色虚线，标记菱形）
# plt.plot(num_malicious, avg_values,
#          linestyle='--', color='#2E8B57', marker='D', markersize=8,
#          linewidth=2, label='BC-MFAIGC')
#
#
# plt.xticks(num_malicious,fontsize=20)
# plt.ylim(-0.2, 1.6)  # 扩展Y轴范围以容纳所有数据
# plt.yticks(fontsize=20)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
#
# plt.xlabel('The proportion of malicious nodes', fontsize=22)
# plt.ylabel('MSE', fontsize=22)
# plt.legend(loc='upper left', frameon=True)
#
#
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig('figure_10.pdf', bbox_inches='tight')
# plt.show()