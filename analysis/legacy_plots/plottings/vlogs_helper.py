import os
import shutil

# 原始文件夹路径，包含多个comm_n文件夹
# source_folder_path = "../logs/flogs/20230801_102059_mnist_bcnn_80"
source_folder_path = "../logs/flogs/20241210_172008_improved_diffusion_model"

# 新的目标文件夹路径，用于存储重命名后的文件
destination_folder_path = "../logs/flogs/$241210_172008_improved_diffusion_model"
if not os.path.exists(destination_folder_path):
    os.mkdir(destination_folder_path)

# a.txt 文件路径
a_txt_path = os.path.join(source_folder_path, "args_used.txt")
destination_a_txt_path = os.path.join(destination_folder_path, "args_used.txt")
shutil.copy(a_txt_path, destination_a_txt_path)

# 获取第一层所有comm_n文件夹的路径
comm_folders = [folder for folder in os.listdir(source_folder_path) if folder.startswith('comm_')]

# 遍历每个comm_n文件夹
for comm_folder in comm_folders:
    # global_comm_n.txt 文件路径
    global_comm_txt_path = os.path.join(source_folder_path, comm_folder, "global_" + comm_folder + ".txt")

    if not os.path.exists(global_comm_txt_path):
        print("global_comm_n.txt not found in {}".format(comm_folder))
        continue
    # 重命名global_comm_n.txt为comm_n.txt并复制到新文件夹
    destination_comm_txt_path = os.path.join(destination_folder_path, comm_folder + ".txt")
    shutil.copy(global_comm_txt_path, destination_comm_txt_path)

print("done")
