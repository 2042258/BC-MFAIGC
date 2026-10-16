import os
import re
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
# 定义文件夹路径
input_folder = str(BCFLDIF_ROOT.joinpath('data_processing', 'tradi', 'outcome'))
output_folder = str(BCFLDIF_ROOT.joinpath('data_processing', 'tradi', 'average_loss'))

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 初始化一个字典来存储每个轮次的损失值
epoch_losses = {}

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # 只处理文件
    if os.path.isfile(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:  # 显式指定编码为 utf-8
                content = file.read()

                # 使用正则表达式提取每个轮次的损失值
                matches = re.findall(r"Epoch (\d+), Test Loss: ([\d.]+)", content)

                for epoch, loss in matches:
                    epoch = int(epoch)
                    loss = float(loss)

                    if epoch not in epoch_losses:
                        epoch_losses[epoch] = []

                    epoch_losses[epoch].append(loss)
        except UnicodeDecodeError:
            print(f"文件 {filename} 的编码不是 utf-8，跳过此文件。")

# 计算每个轮次的平均损失值
average_losses = {}
for epoch, losses in epoch_losses.items():
    average_losses[epoch] = sum(losses) / len(losses)

# 将结果写入输出文件
output_file_path = os.path.join(output_folder, "average_losses.txt")
with open(output_file_path, 'w', encoding='utf-8') as output_file:  # 显式指定编码为 utf-8
    for epoch in sorted(average_losses.keys()):
        output_file.write(f"Epoch {epoch}, Average Test Loss: {average_losses[epoch]}\n")

print(f"平均损失值已写入 {output_file_path}")