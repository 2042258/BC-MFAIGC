import os
import re
import matplotlib.pyplot as plt
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
# 选择要绘制的 num_malicious 值（例如，绘制 num_malicious 为 2 和 3 的线）
selected_malicious_counts = [2, 3]  # 可以根据需要修改
# 定义文件夹路径
folder_path = str(BCFLDIF_ROOT.joinpath('data_processing', 'bcfl', 'processing_outcome'))


# 定义一个函数来解析文件内容并提取数据
def parse_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = {}
    for line in lines:
        match = re.match(r'Comm_(\d+)\s+average\s+B\s+value:\s+([\d.]+)', line)
        if match:
            comm_num = int(match.group(1))
            b_value = float(match.group(2))
            data[comm_num] = b_value

    return data


# 定义一个函数来提取文件名中的恶意数量信息
def extract_malicious_count(file_name):
    # 使用正则表达式提取恶意数量
    match = re.search(r'--num_malicious\s+(\d+)', file_name)
    if match:
        return int(match.group(1))
    return None  # 如果匹配失败，返回 None


# 获取文件夹中的所有文件
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 初始化一个字典来存储不同 num_malicious 的数据
malicious_data = {}

# 遍历每个文件并提取数据
for file in files:
    file_path = os.path.join(folder_path, file)
    data = parse_file(file_path)

    # 提取恶意数量
    malicious_count = extract_malicious_count(file)
    if malicious_count is not None:
        if malicious_count not in malicious_data:
            malicious_data[malicious_count] = {}

        # 合并数据
        for comm_num, b_value in data.items():
            if comm_num not in malicious_data[malicious_count]:
                malicious_data[malicious_count][comm_num] = []
            malicious_data[malicious_count][comm_num].append(b_value)

# 计算每个恶意数量的平均值
for malicious_count, data in malicious_data.items():
    for comm_num in data:
        malicious_data[malicious_count][comm_num] = sum(data[comm_num]) / len(data[comm_num])

# 初始化图表
plt.figure(figsize=(10, 6))



# 遍历每个恶意数量并绘制数据
for malicious_count in selected_malicious_counts:
    if malicious_count in malicious_data:
        data = malicious_data[malicious_count]

        # 提取Comm编号和B值
        comm_nums = sorted(data.keys())
        b_values = [data[comm] for comm in comm_nums]

        # 绘制数据
        plt.plot(comm_nums, b_values, label=f'num_malicious {malicious_count}')

# 设置图表标题和标签
plt.title('Average B Value vs Comm Number')
plt.xlabel('Comm Number')
plt.ylabel('Average B Value')
plt.legend()
plt.grid(True)

# 显示图表
plt.show()