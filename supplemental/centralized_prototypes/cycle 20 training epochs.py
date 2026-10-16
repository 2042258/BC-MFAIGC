import os
import sys
import argparse
import random
import time
from datetime import datetime
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from BFL.DatasetLoad import DatasetLoad
from models.Models import ImprovedDiffusionModel

for i in range(1):
    print(f"循环次数: {i + 1}")
    # 设置程序执行时间用于日志记录
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_files_folder_path = f"../logs/nflogs/{date_time}"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Diffusion Model Training")

    # 训练参数
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('-B', '--batchsize', type=int, default=16, help='local train batch size')
    parser.add_argument('-mn', '--model_name', type=str, default='improved_diffusion_model', help='the model to train')
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.001, help="learning rate, use value from origin paper as default")
    parser.add_argument('-op', '--optimizer', type=str, default="Adam", help='optimizer to be used, by default implementing Adam')
    parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to devices')
    parser.add_argument('-max_ncomm', '--max_num_comm', type=int, default=20, help='maximum number of communication rounds, may terminate early if converges')
    parser.add_argument('-nd', '--num_devices', type=int, default=10, help='numer of the devices in the simulation network')
    parser.add_argument('-st', '--shard_test_data', type=int, default=0, help='it is easy to see the global models are consistent across devices when the test dataset is NOT sharded')
    parser.add_argument('-nm', '--num_malicious', type=int, default=4, help="number of malicious nodes in the network. malicious node's data sets will be introduced Gaussian noise")
    parser.add_argument('-nv', '--noise_variance', type=int, default=2, help="noise variance level of the injected Gaussian Noise")
    parser.add_argument('-le', '--default_local_epochs', type=int, default=5, help='local train epoch. Train local model by this same num of epochs for each worker, if -mt is not specified')

    # 添加恶意客户端加噪函数
    def malicious_worker_add_noise_to_weights(net, noise_variance, dev):
        with torch.no_grad():
            for m in net.modules():
                if hasattr(m, 'weight'):
                    noise = noise_variance * torch.randn(m.weight.size())
                    m.weight.add_(noise.to(dev))

    if __name__ == "__main__":
        # 获取参数
        args = parser.parse_args()
        args = args.__dict__

        # 检测CUDA
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # 创建日志文件夹
        if not os.path.exists('../logs'):
            os.makedirs('../logs')
        if not os.path.exists('../logs/nflogs'):
            os.makedirs('../logs/nflogs')
        os.mkdir(log_files_folder_path)

        # 保存使用的参数
        with open(f'{log_files_folder_path}/args_used.txt', 'w') as f:
            f.write("Command line arguments used -\n")
            f.write(' '.join(sys.argv[1:]))
            f.write("\n\nAll arguments used -\n")
            for arg_name, arg in args.items():
                f.write(f'\n--{arg_name} {arg}')

        # 创建神经网络模型
        net = ImprovedDiffusionModel()
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
        print(f"{torch.cuda.device_count()} GPUs are available to use!")
        net = net.to(dev)

        # 设置损失函数和优化器
        loss_func = F.mse_loss
        if args['optimizer'] == "SGD":
            opti = optim.SGD(net.parameters(), lr=args['learning_rate'])
        elif args['optimizer'] == "Adam":
            opti = optim.Adam(net.parameters(), lr=args['learning_rate'], weight_decay=1e-5)

        # 加载数据集
        cifar10_dataset = DatasetLoad('cifar10', args['IID'])
        train_data = cifar10_dataset.train_data
        train_label = cifar10_dataset.train_label
        test_data = cifar10_dataset.test_data
        test_label = cifar10_dataset.test_label

        train_data_loader = DataLoader(TensorDataset(torch.tensor(train_data), torch.tensor(train_label)), batch_size=args['batchsize'], shuffle=True)
        test_data_loader = DataLoader(TensorDataset(torch.tensor(test_data), torch.tensor(test_label)), batch_size=args['batchsize'], shuffle=False)

        # 训练循环
        for epoch in range(args['max_num_comm']):
            # 判断是否触发攻击
            attack_triggered = random.random() < 0.6  # 40%的概率触发攻击

            if attack_triggered:
                print(f"Epoch {epoch + 1}: Attack triggered! Adding noise to model weights.")
                # 对模型权重添加噪声
                malicious_worker_add_noise_to_weights(net, args['noise_variance'], dev)
            else:
                print(f"Epoch {epoch + 1}: No attack.")

            net.train()
            for i, (data, _) in enumerate(train_data_loader):  # 忽略标签
                data = data.to(dev)
                opti.zero_grad()
                preds = net(data)
                loss = loss_func(preds, data)
                loss.backward()
                opti.step()

            # 验证模型
            net.eval()
            total_loss = 0
            num_batches = 0
            with torch.no_grad():
                for data, _ in test_data_loader:  # 忽略标签
                    data = data.to(dev)
                    preds = net(data)
                    loss = loss_func(preds, data)
                    total_loss += loss.item()
                    num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{args['max_num_comm']}, Test Loss: {avg_loss}")

            # 记录日志
            with open(f"{log_files_folder_path}/accuracy_comm_{epoch + 1}.txt", "a") as file:
                file.write(f"Epoch {epoch + 1}, Test Loss: {avg_loss}\n")

        # 结束时间
        end_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'{log_files_folder_path}/time.txt', 'a') as f:
            f.write("\nend time: " + end_time)