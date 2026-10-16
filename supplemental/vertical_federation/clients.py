import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
import copy
from torch import optim, nn
import random

"""单个客户端
	恶意工作者给权重增加噪声
	重置噪声的方差variance
	模型本地更新
	评估模型的准确率

Returns:
	_type_: _description_
"""
class client(object):
	def __init__(self, idx, is_malicious, noise_variance, trainDataSet, assigned_test_dl, learning_rate, net, dev):
		self.idx = idx
		self.is_malicious = is_malicious
		self.noise_variance = noise_variance
		self.variance_of_noises = None or []
		self.train_ds = trainDataSet
		self.test_dl = assigned_test_dl
		self.dev = dev
		self.net = copy.deepcopy(net)
		self.opti = optim.SGD(self.net.parameters(), lr=learning_rate)
		self.train_dl = None
		self.local_parameters = None


	def malicious_worker_add_noise_to_weights(self, m):
		with torch.no_grad():
			if hasattr(m, 'weight'):
				noise = self.noise_variance * torch.randn(m.weight.size())
				variance_of_noise = torch.var(noise)
				m.weight.add_(noise.to(self.dev))
				self.variance_of_noises.append(float(variance_of_noise))

	def reset_variance_of_noise(self):
		self.variance_of_noises.clear()

	def localUpdate(self, localEpoch, localBatchSize, lossFun, global_parameters, comm_round_folder, i):
		self.net.load_state_dict(global_parameters, strict=True)
		self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)

		# 梯度累积步数
		accumulation_steps = 4

		for epoch in range(localEpoch):
			for i, (data, _) in enumerate(self.train_dl):  # 忽略标签
				data = data.to(self.dev)
				t = torch.randint(0, 1000, (data.size(0),), device=self.dev).long()  # 随机生成时间步
				preds = self.net(data, t)  # 传递时间步 t

				# 确保 preds 和 data 的形状匹配
				if preds.shape != data.shape:
					raise ValueError(f"Shape mismatch: preds {preds.shape} != data {data.shape}")

				# 使用均方误差损失
				loss_func = nn.MSELoss()
				loss = loss_func(preds, data)
				loss = loss / accumulation_steps
				loss.backward()

				# 梯度裁剪
				torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

				if (i + 1) % accumulation_steps == 0:
					self.opti.step()
					self.opti.zero_grad()

			# 在每轮训练后，记录是否恶意节点，以及本轮的损失值
			with open(f"{comm_round_folder}/{self.idx}_local_comm_{i + 1}.txt", "a") as file:
				is_malicious_node = "M" if self.is_malicious else "B"
				loss_this_epoch = self.evaluate_model_weights(self.net.state_dict())
				if np.isnan(loss_this_epoch):
					loss_this_epoch = "nan"
				file.write(f"{self.idx} {is_malicious_node} epoch_{epoch + 1}: {loss_this_epoch}\n")

		# 如果是恶意节点，给权重加上噪声后，再重新评估模型损失
		if self.is_malicious:
			self.net.apply(self.malicious_worker_add_noise_to_weights)
			print(f"malicious client {self.idx} has added noise to its local updated weights before transmitting")
			with open(f"{comm_round_folder}/{self.idx}_{is_malicious_node}_local_comm_{i + 1}.txt", "a") as file:
				file.write(
					f"{self.idx} {is_malicious_node} noise_injected: {self.evaluate_model_weights(self.net.state_dict())}\n")
				file.write(f"{self.idx} {is_malicious_node} noise_variances: {self.variance_of_noises}\n")

		return self.net.state_dict()

	def evaluate_model_weights(self, global_parameters):
		with torch.no_grad():
			self.net.load_state_dict(global_parameters, strict=True)
			total_loss = 0
			num_batches = 0
			for data, _ in self.test_dl:  # 忽略标签
				data = data.to(self.dev)
				t = torch.randint(0, 1000, (data.size(0),), device=self.dev).long()  # 随机生成时间步
				preds = self.net(data, t)  # 传递时间步 t

				# 确保 preds 和 data 的形状匹配
				if preds.shape != data.shape:
					raise ValueError(f"Shape mismatch: preds {preds.shape} != data {data.shape}")

				# 使用均方误差计算损失
				loss = nn.MSELoss()(preds, data)
				total_loss += loss.item()
				num_batches += 1

			return total_loss / num_batches

"""客户端群组
	平均分配数据集
"""
class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, learning_rate, dev, net, num_malicious, noise_variance, shard_test_data):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.net = net
        self.learning_rate = learning_rate
        self.dev = dev
        self.clients_set = {}
        self.num_malicious = num_malicious
        self.noise_variance = noise_variance
        self.shard_test_data = shard_test_data

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        """
        随机采样生成恶意节点，给客户端分配数据集
        """
        mnist_dataset = GetDataSet(self.data_set_name, self.is_iid)

        # perpare training data
        # 准备训练数据
        train_data = mnist_dataset.train_data
        train_label = mnist_dataset.train_label
        # shard dataset and distribute among clients
        # shard train
        # 分片数据集
        shard_size_train = mnist_dataset.train_data_size // self.num_of_clients // 2  # todo：为什么除以2
        shards_id_train = np.random.permutation(mnist_dataset.train_data_size // shard_size_train)

        # perpare test data
        # 准备测试数据
        if not self.shard_test_data:  # todo：如果不是分片测试数据，就... 不知道有啥区别
            test_data = torch.tensor(mnist_dataset.test_data)
            test_label = torch.argmax(torch.tensor(mnist_dataset.test_label), dim=1)
            test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
        else:
            test_data = mnist_dataset.test_data
            test_label = mnist_dataset.test_label
            # shard test
            shard_size_test = mnist_dataset.test_data_size // self.num_of_clients // 2
            shards_id_test = np.random.permutation(mnist_dataset.test_data_size // shard_size_test)

        # 随机采样恶意节点
        malicious_nodes_set = []
        if self.num_malicious:
            malicious_nodes_set = random.sample(range(self.num_of_clients), self.num_malicious)

        for i in range(self.num_of_clients):
            is_malicious = False
            # make it more random by introducing two shards
            shards_id_train1 = shards_id_train[i * 2]
            shards_id_train2 = shards_id_train[i * 2 + 1]
            # distribute training data
            data_shards1 = train_data[shards_id_train1 * shard_size_train: shards_id_train1 * shard_size_train + shard_size_train]
            data_shards2 = train_data[shards_id_train2 * shard_size_train: shards_id_train2 * shard_size_train + shard_size_train]
            label_shards1 = train_label[shards_id_train1 * shard_size_train: shards_id_train1 * shard_size_train + shard_size_train]
            label_shards2 = train_label[shards_id_train2 * shard_size_train: shards_id_train2 * shard_size_train + shard_size_train]
            local_train_data, local_train_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_train_label = np.argmax(local_train_label, axis=1)
            # distribute test data
            if self.shard_test_data:
                shards_id_test1 = shards_id_test[i * 2]
                shards_id_test2 = shards_id_test[i * 2 + 1]
                data_shards1 = test_data[shards_id_test1 * shard_size_test: shards_id_test1 * shard_size_test + shard_size_test]
                data_shards2 = test_data[shards_id_test2 * shard_size_test: shards_id_test2 * shard_size_test + shard_size_test]
                label_shards1 = test_label[shards_id_test1 * shard_size_test: shards_id_test1 * shard_size_test + shard_size_test]
                label_shards2 = test_label[shards_id_test2 * shard_size_test: shards_id_test2 * shard_size_test + shard_size_test]
                local_test_data, local_test_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
                local_test_label = torch.argmax(torch.tensor(local_test_label), dim=1)
                test_data_loader = DataLoader(TensorDataset(torch.tensor(local_test_data), torch.tensor(local_test_label)), batch_size=100, shuffle=False)
            # assign data to a client and put in the clients set
            if i in malicious_nodes_set:
                is_malicious = True
                # add Gussian Noise

            # 实例化客户端
            client_idx = f'client_{i+1}'
            someone = client(client_idx, is_malicious, self.noise_variance, TensorDataset(torch.tensor(local_train_data), torch.tensor(local_train_label)), test_data_loader, self.learning_rate, self.net, self.dev)
            self.clients_set[client_idx] = someone



