import os
from pathlib import Path

BCFLDIF_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "BFL" / "main.py").is_file()
)
# 定义 nflogs 文件夹路径
nflogs_path = str(BCFLDIF_ROOT.joinpath('logs', 'nflogs'))

# 定义输出文件夹路径
output_folder = str(BCFLDIF_ROOT.joinpath('data_processing', 'tradi', 'outcome'))

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历 nflogs 文件夹下的所有时间戳文件夹
for timestamp_folder in os.listdir(nflogs_path):
    timestamp_folder_path = os.path.join(nflogs_path, timestamp_folder)

    # 检查是否是文件夹
    if os.path.isdir(timestamp_folder_path):
        # 定义输出文件名
        output_file = os.path.join(output_folder, f"{timestamp_folder}_outcome")

        # 打开输出文件以写入内容
        with open(output_file, "w") as outfile:
            # 遍历1到20的文件名
            for i in range(1, 101):
                file_name = f"accuracy_comm_{i}.txt"
                file_path = os.path.join(timestamp_folder_path, file_name)

                # 检查文件是否存在
                if os.path.exists(file_path):
                    # 打开并读取文件内容
                    with open(file_path, "r") as infile:
                        content = infile.read()
                        # 将内容写入输出文件
                        outfile.write(f"{file_name}:\n{content}\n\n")
                else:
                    print(f"文件 {file_path} 不存在")

print("所有内容已提取并保存到指定文件夹。")