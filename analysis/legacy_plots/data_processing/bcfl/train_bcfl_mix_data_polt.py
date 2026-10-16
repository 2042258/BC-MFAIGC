import os
import re
import matplotlib.pyplot as plt
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
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


# 定义一个函数来提取文件名中的日期时间和恶意数量信息
def extract_legend_info(file_name):
    # 使用正则表达式提取日期时间和恶意数量
    match = re.match(r'(\d{8}_\d{6})_bcfl_num_malicious_--num_malicious\s+(\d+)\.txt', file_name)
    if match:
        date_time = match.group(1)  # 提取日期时间部分
        malicious_count = match.group(2)  # 提取恶意数量部分
        return f"{date_time} (num_malicious {malicious_count})"
    return file_name  # 如果匹配失败，返回原始文件名


# 获取文件夹中的所有文件
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 初始化图表
plt.figure(figsize=(10, 6))

# 遍历每个文件并绘制数据
for file in files:
    file_path = os.path.join(folder_path, file)
    data = parse_file(file_path)

    # 提取Comm编号和B值
    comm_nums = sorted(data.keys())
    b_values = [data[comm] for comm in comm_nums]

    # 提取图例信息
    legend_info = extract_legend_info(file)

    # 绘制数据
    plt.plot(comm_nums, b_values, label=legend_info)

# 设置图表标题和标签
plt.title('Average B Value vs Comm Number')
plt.xlabel('Comm Number')
plt.ylabel('Average B Value')
plt.legend()
plt.grid(True)

# 显示图表
plt.show()