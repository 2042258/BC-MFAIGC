import matplotlib.pyplot as plt
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
# 定义文件路径
file_path = str(BCFLDIF_ROOT.joinpath('data_processing', 'tradi', 'average_loss', 'average_losses.txt'))

# 初始化列表存储轮次和损失值
epochs = []
losses = []

# 读取文件内容
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每一行的内容
        parts = line.strip().split(", ")
        if len(parts) == 2:
            epoch = int(parts[0].split(":")[0].split(" ")[1])  # 提取轮次
            loss = float(parts[1].split(":")[1])  # 提取损失值
            epochs.append(epoch)
            losses.append(loss)

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label=' Test Loss')

# 设置图表标题和标签
plt.title("Average Test Loss vs Epoch", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Average Test Loss", fontsize=14)

# 设置网格
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图例
plt.legend(fontsize=12)

# 显示图表
plt.tight_layout()
plt.show()