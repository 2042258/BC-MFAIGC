# figure 1

import matplotlib.pyplot as plt
from os import listdir
import sys


if len(sys.argv) < 5:
    print("Usage: python script.py <log_folder_b> <log_folder_m> <log_folder_PoS_3_vh_008_run1> <log_folder_PoS_BCNN>")
    sys.exit(1)


log_folder_b = sys.argv[1]  # benign_devices 良性设备
log_folder_m = sys.argv[2]  # malicious_devices 恶意设备	
log_folder_PoS_3_vh_008_run1 = sys.argv[3]
log_folder_PoS_BCNN = sys.argv[4]

all_rounds_log_files_b = sorted([f for f in listdir(log_folder_b) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_log_files_m = sorted([f for f in listdir(log_folder_m) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_PoS_3_vh_008_run1 = sorted([f for f in listdir(log_folder_PoS_3_vh_008_run1) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_PoS_BCNN = sorted([f for f in listdir(log_folder_PoS_BCNN) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))

draw_comm_rounds = min(len(all_rounds_log_files_b), len(all_rounds_log_files_m), len(all_rounds_PoS_3_vh_008_run1), len(all_rounds_PoS_BCNN))
plt.figure(dpi=250)

# get num of devices with their maliciousness
benign_devices_idx_list = []
malicious_devices_idx_list = []
comm_1_file_path = f"{log_folder_m}/comm_1.txt"
file = open(comm_1_file_path,"r") 
log_whole_text = file.read() 
lines_list = log_whole_text.split("\n")
for line in lines_list:
	if line.startswith('client'):
		device_idx = line.split(":")[0].split(" ")[0]
		device_maliciousness = line.split(":")[0].split(" ")[-1]
		if device_maliciousness == 'M':
			malicious_devices_idx_list.append(device_idx)
		else:
			benign_devices_idx_list.append(device_idx)

total_malicious_devices = len(malicious_devices_idx_list)
total_devices = len(malicious_devices_idx_list + benign_devices_idx_list)
	
b_device_accuracies_across_rounds = []
m_device_accuracies_across_rounds = []
PoS_3_vh_008_run1_accuracies = []
PoS_BCNN_accuracies = []

for log_file in all_rounds_log_files_b:
	file = open(f"{log_folder_b}/{log_file}","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	for line in lines_list:
		if line.startswith('client_1'):
			accuracy = round(float(line.split(":")[-1]), 3)
			b_device_accuracies_across_rounds.append(accuracy)
			break

for log_file in all_rounds_log_files_m:
	file = open(f"{log_folder_m}/{log_file}","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	for line in lines_list:
		if line.startswith('client_1'):
			accuracy = round(float(line.split(":")[-1]), 3)
			m_device_accuracies_across_rounds.append(accuracy)
			break

# get global accuracy PoS_3_vh_008
runs = ['PoS_3_vh_008_run1', 'PoS_BCNN']
for run in runs:
	vars()[f"all_rounds_{run}"]
	for log_file_folder in vars()[f"all_rounds_{run}"]:
		if len(log_file_folder) > 8:
			continue
		try:
			file = open(f"{vars()[f'log_folder_{run}']}/{log_file_folder}/accuracy_{log_file_folder}.txt","r")
		except:
			log_file_folder = f"comm_{int(log_file_folder.split('_')[-1])-1}"
			file = open(f"{vars()[f'log_folder_{run}']}/{log_file_folder}/accuracy_{log_file_folder}.txt","r")
		log_whole_text = file.read() 
		lines_list = log_whole_text.split("\n")
		for line in lines_list:
			if line.startswith('device_1'):
				accuracy = round(float(line.split(":")[-1]), 3)
				vars()[f'{run}_accuracies'].append(accuracy)
				break

m_device_accuracies_across_rounds = m_device_accuracies_across_rounds[:draw_comm_rounds]
b_device_accuracies_across_rounds = b_device_accuracies_across_rounds[:draw_comm_rounds]
PoS_3_vh_008_run1_accuracies = PoS_3_vh_008_run1_accuracies[:draw_comm_rounds]
PoS_BCNN_accuracies = PoS_BCNN_accuracies[:draw_comm_rounds]

# draw graphs over all available comm rounds
plt.plot(range(draw_comm_rounds), b_device_accuracies_across_rounds, label=f'Vanilla FL all {total_devices} legitimate devices', color='orange')
plt.plot(range(draw_comm_rounds), PoS_3_vh_008_run1_accuracies, label=r'VBFL_CNN 3 out of 20 malicious devices', color='green')
plt.plot(range(draw_comm_rounds), PoS_BCNN_accuracies, label=r'VBFL_BCNN 3 out of 20 malicious devices', color='red')
plt.plot(range(draw_comm_rounds), m_device_accuracies_across_rounds, label=f'Vanilla FL {total_malicious_devices} out of {total_devices} malicious devices', color='blue')

# 每隔20个下标，打上准确率的注释
if b_device_accuracies_across_rounds:
	annotating_points = 5
	skipped_1 = False
	for accuracy_iter in range(len(b_device_accuracies_across_rounds)): 
		if not accuracy_iter % (len(b_device_accuracies_across_rounds) // annotating_points):  # 100//5=20, 判断下标是否是20的倍数
			if not skipped_1:  # 跳过第一个
				skipped_1 = True
				continue
			plt.annotate(b_device_accuracies_across_rounds[accuracy_iter], xy=(accuracy_iter, b_device_accuracies_across_rounds[accuracy_iter]), size=12)  # 打上准确率的注释

if m_device_accuracies_across_rounds:
	annotating_points = 5
	skipped_1 = False
	for accuracy_iter in range(len(m_device_accuracies_across_rounds)):
		if not accuracy_iter % (len(m_device_accuracies_across_rounds) // annotating_points):
			if not skipped_1:
				skipped_1 = True
				continue
			plt.annotate(m_device_accuracies_across_rounds[accuracy_iter], xy=(accuracy_iter, m_device_accuracies_across_rounds[accuracy_iter]), size=12)

if PoS_3_vh_008_run1_accuracies:
	annotating_points = 5
	skipped_1 = False
	for accuracy_iter in range(len(PoS_3_vh_008_run1_accuracies)):
		if not accuracy_iter % (len(PoS_3_vh_008_run1_accuracies) // annotating_points):
			if not skipped_1:
				skipped_1 = True
				continue
			plt.annotate(PoS_3_vh_008_run1_accuracies[accuracy_iter], xy=(accuracy_iter, PoS_3_vh_008_run1_accuracies[accuracy_iter]), size=12)

if PoS_BCNN_accuracies:
	annotating_points = 5
	skipped_1 = False
	for accuracy_iter in range(len(PoS_BCNN_accuracies)):
		if not accuracy_iter % (len(PoS_BCNN_accuracies) // annotating_points):
			if not skipped_1:
				skipped_1 = True
				continue
			plt.annotate(PoS_BCNN_accuracies[accuracy_iter], xy=(accuracy_iter, PoS_BCNN_accuracies[accuracy_iter]), size=12)

plt.legend(loc='best', bbox_to_anchor=(0.32,0.7))
plt.xlabel('Communication Round')
plt.ylabel('Global Accuracy')
plt.title('Comparison of Global Model Accuracy')
# plt.title('Global Model Accuracy Comparisons Before and After Introducing Noices through vanilla FedAvg Communication Rounds On MNIST Dataset Using MNIST_CNN')
plt.show()
print()