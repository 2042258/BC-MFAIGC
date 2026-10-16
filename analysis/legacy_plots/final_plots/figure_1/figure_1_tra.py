import os
import re
import matplotlib.pyplot as plt
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
nflogs_path = str(BCFLDIF_ROOT.joinpath('logs', 'nflogs'))
specified_num_malicious_values = [0, 1, 2, 3, 4, 5, 6]
num_malicious_losses = {}

# 遍历文件夹并收集数据
for timestamp_folder in os.listdir(nflogs_path):
    folder_path = os.path.join(nflogs_path, timestamp_folder)
    if not os.path.isdir(folder_path):
        continue

    # 提取 --num_malicious 参数
    args_file = os.path.join(folder_path, "args_used.txt")
    num_malicious = None
    if os.path.exists(args_file):
        with open(args_file, 'r', encoding='utf-8') as f:
            match = re.search(r"--num_malicious (\d+)", f.read())
            if match:
                num_malicious = int(match.group(1))
    if num_malicious not in specified_num_malicious_values:
        continue

    # 处理当前文件夹的损失数据
    epoch_losses = {}
    for comm_round in range(1, 101):
        file_path = os.path.join(folder_path, f"accuracy_comm_{comm_round}.txt")
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r') as f:
            matches = re.findall(r"Epoch (\d+), Test Loss: ([\d.]+)", f.read())
            for epoch_str, loss_str in matches:
                epoch = int(epoch_str)
                loss = float(loss_str)
                if epoch not in epoch_losses:
                    epoch_losses[epoch] = []
                epoch_losses[epoch].append(loss)

    # 合并到全局字典
    for epoch, losses in epoch_losses.items():
        if epoch not in num_malicious_losses.get(num_malicious, {}):
            num_malicious_losses.setdefault(num_malicious, {})[epoch] = []
        num_malicious_losses[num_malicious][epoch].extend(losses)

# 计算平均值
average_losses = {}
for num_malicious, epochs in num_malicious_losses.items():
    average_losses[num_malicious] = {
        epoch: sum(losses) / len(losses) for epoch, losses in epochs.items()
    }

# 打印第一百轮的损失值
print("第一百轮的平均测试损失值：")
for num in specified_num_malicious_values:
    if num in average_losses and 100 in average_losses[num]:
        loss = average_losses[num][100]
        print(f"--num_malicious {num}: {loss:.4f}")
    else:
        print(f"--num_malicious {num}: 数据缺失")

# 绘制图表
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for idx, num in enumerate(specified_num_malicious_values):
    if num in average_losses:
        epochs = sorted(average_losses[num].keys())
        losses = [average_losses[num][e] for e in epochs]
        plt.plot(epochs, losses, marker='o', linestyle='-',
                 color=colors[idx % len(colors)], label=f'num_malicious={num}')

plt.title("Average Test Loss per Epoch", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()