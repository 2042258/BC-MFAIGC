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

# 设置图表标题和标签
plt.title("Average Test Loss vs Epoch for Specified --num_malicious Values", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Average Test Loss", fontsize=14)

# 设置网格
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图例
plt.legend(fontsize=12)

# 显示图表
plt.tight_layout()
plt.show()