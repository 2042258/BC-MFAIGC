import os
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
# 定义根目录
root_dir = str(BCFLDIF_ROOT.joinpath('logs', 'blogs'))  # 修改为实际的根目录
output_dir = str(BCFLDIF_ROOT.joinpath('data_processing', 'bcfl', 'processing_outcome'))

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历blogs目录下所有时间戳文件夹
for timestamp_dir in os.listdir(root_dir):
    timestamp_path = os.path.join(root_dir, timestamp_dir)

    # 检查是否是时间戳文件夹
    if not os.path.isdir(timestamp_path):
        continue

    # 从args_used.txt中提取--num_malicious的值
    args_file = os.path.join(timestamp_path, 'args_used.txt')
    num_malicious = None
    if os.path.exists(args_file):
        with open(args_file, 'r') as file:
            for line in file:
                if '--num_malicious' in line:
                    num_malicious = line.split(':')[-1].strip()
                    break

    # 检查是否成功提取--num_malicious
    if num_malicious is None:
        print(f"Warning: --num_malicious not found in {args_file}. Skipping {timestamp_dir}.")
        continue

    # 构建输出文件名
    output_file = os.path.join(output_dir, f'{timestamp_dir}_bcfl_num_malicious_{num_malicious}.txt')

    # 初始化一个字典来存储每个comm的B:后的数值
    comm_b_values = {}

    # 遍历comm_1到comm_20目录
    for comm_num in range(1, 101):  # 修改为21以包含comm_20
        comm_dir = os.path.join(timestamp_path, f'comm_{comm_num}')
        file_path = os.path.join(comm_dir, f'worker_final_local_accuracies_comm_{comm_num}.txt')  # 添加.txt后缀

        print(f"Checking file: {file_path}")  # 打印文件路径
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")  # 打印未找到的文件
            continue

        # 初始化当前comm的B:后的数值列表
        b_values = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if "worker B" in line:
                    value = float(line.split(":")[-1].strip())
                    b_values.append(value)

        # 计算当前comm的B:后数值的平均值
        if b_values:
            average_b_value = sum(b_values) / len(b_values)
            comm_b_values[comm_num] = average_b_value
            print(f"Comm_{comm_num} average B value: {average_b_value}")
        else:
            print(f"No B values found in Comm_{comm_num}")

    # 将所有comm的结果写入汇总文件
    with open(output_file, 'w') as outfile:
        for comm_num, average_b_value in comm_b_values.items():
            outfile.write(f"Comm_{comm_num} average B value: {average_b_value}\n")

    print(f"All results for {timestamp_dir} saved to: {output_file}")