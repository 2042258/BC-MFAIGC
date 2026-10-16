# 针对传统联邦学习聚合算法中各个参与者缺乏模型场景化的问题，
# 本文提出的基于Ｅｕｃｌｉｄ距离的联邦学习聚合算法也对其进行了改进。
# 参与者通过式（２）在本地聚合生一个新的模型并继续下一步的训练，实现模型的本地化。
# ｗ～ｔ＋１ｉ＝ΣＮｉ＝１ｋｉ，ｊｗｔ＋１ｊ，　ｋｉ，ｊ＝ｅα（ｗｔｊ－ｗｔｉ）■２ΣＮｌ＝１ｅα（ｗｔｊ－ｗｔｉ）■２.（２）

# 假设我们已经从日志中提取了良性节点和恶意节点的更新数据
benign_updates = [
    (0.014123712995648383, 0.013844165907800198),  # device_4 by device_6
    (0.014123712995648383, 0.013788637931644917),  # device_9 by device_6
    (0.015743837422132492, 0.013788637955486774),  # device_9 by device_10
    (0.015743837422132492, 0.013844165931642055),  # device_4 by device_10
    (0.015815407583117485, 0.01378863794207573),   # device_9 by device_7
    (0.015815407583117485, 0.013844165891408921),  # device_4 by device_7
    (0.014900561754405498, 0.013788637951016426),  # device_9 by device_8
    (0.014900561754405498, 0.013844165903329849),  # device_4 by device_8
]

malicious_updates = [
    (0.014123712995648383, 1.4118836574554443),  # device_5 by device_6
    (0.015743837422132492, 1.4118836574554443),  # device_5 by device_10
    (0.015815407583117485, 1.4118836574554443),  # device_5 by device_7
    (0.014900561754405498, 1.4118836574554443),  # device_5 by device_8
]

# 计算良性节点更新的平均提升
benign_improvements = [before - after for before, after in benign_updates]
average_benign_improvement = sum(benign_improvements) / len(benign_improvements)

# 计算恶意节点更新的平均降低
malicious_deteriorations = [before - after for before, after in malicious_updates]
average_malicious_deterioration = sum(malicious_deteriorations) / len(malicious_deteriorations)

# 确定 validator_threshold
validator_threshold = average_benign_improvement - average_malicious_deterioration

print(f"Calculated validator_threshold: {validator_threshold}")