# BCFLDIF 二修实验阶段 0：实现、日志与图表审计

> 审计日期：2026-09-24
> 
> 本报告只记录阶段 0 的只读核查结果；没有修改训练代码、模型代码或已有日志，也没有把任何数值当作新的实验结果。

## 1. 阶段结论

当前实现**不是标准 DDPM**，因此现有 MSE 曲线不能作为生成质量指标。正式实验暂不放行；应先完成指标/模型路线决策、日志字段设计和小规模试跑。

已有 `logs/blogs` 可以用于核对旧实现、校准攻击强度和检查日志口径，但由于：

- 代码仓库尚无初始 commit，旧日志无法绑定到可核验的代码版本；
- 运行参数没有记录随机种子；
- 旧攻击与阈值之间存在明显可分离间隔；
- 图表脚本混用了不同日志格式，并有硬编码结果；

因此旧日志不应直接作为二修的正式重复实验或新结论依据。

## 2. 实现核查

### 2.1 模型路径不是标准 DDPM

实际模型文件是 `models/Models.py`，由 `BFL/main.py` 导入。当前 `DDPM` 只有时间嵌入和一次 U-Net 前向传播：

- 没有 beta/alpha 噪声调度；
- 没有前向加噪 `q_sample`；
- 没有反向采样 `p_sample` 或采样循环；
- 网络输出的是与输入图像同形状的结果，不是噪声 `epsilon`；
- 没有生成样本保存或采样评估路径。

训练实际执行于 `BFL/Device.py:786-845`：

```text
随机生成 t -> 将未加噪的 data 输入模型 -> MSE(preds, data)
```

因此当前损失是输入—输出重构误差，不是 DDPM 噪声预测损失，也不是生成样本与真实图像之间的 MSE。

### 2.2 当前训练/验证指标

- `worker_local_update()` 内部实际使用 `nn.MSELoss()`；`main.py` 传入的 `F.cross_entropy` 没有生效。
- `validate_model_weights()` 在测试集上计算平均 batch MSE，但每个 batch 都重新随机采样 `t`，同一权重的评估结果不完全可复现。
- 验证函数和日志仍把该值称为 `accuracy`，应改为明确的 loss/discrepancy 名称。
- 梯度累积步数为 4；每个 epoch 末尾不足 4 个 batch 的梯度没有显式提交，正式实验前应决定修正或记录这一训练预算差异。

### 2.3 数据集和 IID 含义

`BFL/DatasetLoad.py` 只读取 `data/cifar-10-batches-py/data_batch_1`，即 10,000 张训练图像，而不是 CIFAR-10 的全部 50,000 张训练图像；测试集读取完整 `test_batch`。

当前数据划分：

- `IID=1`：训练样本先随机打乱，再按 shard 分配；
- `IID=0`：按标签排序后再按 shard 分配，属于标签排序式非 IID，不是 Dirichlet 划分；
- 10 个节点时每个节点得到两个 500 张训练 shard；
- `shard_test_data=0` 时每个节点共享完整测试集，不能直接作为“各节点本地异质验证集”的实验。

DDPM 正式实验还需冻结输入归一化协议。当前输入范围为 `[0,1]`，而模型末端使用 `Tanh()`；标准 DDPM 通常需要明确的 `[−1,1]` 数据预处理。

### 2.4 随机性与角色分配

仓库内没有发现 `random.seed`、`numpy.random.seed` 或 `torch.manual_seed`。随机性来自：

- 训练数据 DataLoader shuffle；
- NumPy shard 排列；
- Python 随机抽取恶意节点和角色；
- 每个验证 batch 的随机时间步；
- 每轮随机角色重分配。

`--hard_assign *,*,*` 并不是固定角色实验：代码只保证至少一个 worker、validator 和 miner，其余节点每轮随机分配。恶意节点是在全体设备中抽取的，因此每轮真正作为 worker 提交恶意更新的节点数会变化。

正式实验应记录 seed、恶意节点集合、每轮角色集合和实际提交的 worker 数量；若要比较攻击比例，建议固定角色配置，或者明确将比例定义为全体网络节点比例而非每轮参与 worker 比例。

## 3. 攻击和筛选核查

### 3.1 攻击位置和参数

`BFL/Device.py:775-783, 824-833` 表明：

- 节点完成本地训练后，在发送更新前给模型权重加噪声；
- 只修改带有 `.weight` 属性的层，不修改 bias；
- 实际噪声为 `noise_variance * torch.randn(...)`；
- 参数名为 `noise_variance`，但 `-nv 2` 实际更接近标准差系数 2，噪声方差约为 4，而不是方差 2；
- 当前实验的恶意行为是固定强度的本地模型权重高斯扰动，不是数据投毒，也不是在线自适应攻击。

### 3.2 筛选规则

`BFL/Device.py:1494-1510` 实际判定分数为：

```text
score = worker_update_MSE - validator_local_MSE
reject/filter if score > validator_threshold
```

当前默认阈值是 `0.1`。这不是论文中写出的绝对差值规则，正式实验必须统一代码、算法和公式。

### 3.3 旧日志中的可分离性

对 22 个完整旧 `logs/blogs` 运行的逐验证记录重新解析后：

- 共 22,243 条验证记录；
- 恶意更新被拒绝 6,242 条；
- 恶意更新被接受 0 条；
- 良性更新被拒绝 0 条；
- 良性更新被接受 16,001 条；
- 恶意分数的全局最小值约为 `0.5316`；
- 良性分数的全局最大值约为 `0.002244`；
- 使用阈值 `0.1` 时两类分数存在很大的间隔。

这支持审稿人关于旧攻击“容易分离”的判断，不能作为近阈值鲁棒性证据。

### 3.4 旧日志的混淆矩阵命名错误

`process_block()` 中的计数命名与标准“恶意为 positive”的定义相反：

| 代码写入的名称 | 代码实际统计 | 标准含义 |
|---|---|---|
| `TP` | 良性更新被接受 | TN |
| `TN` | 恶意更新被拒绝 | TP |
| `FP` | 恶意更新被接受 | FN |
| `FN` | 良性更新被拒绝 | FP |

根目录文件名也存在同样的语义混乱，例如 `true_negative_malicious_nodes_inside_caught.txt` 实际记录的是恶意节点被捕获。二修必须从逐条验证记录和最终聚合决定重新计算，不能沿用这些名称或旧汇总数字。

## 4. 旧日志和图表核查

### 4.1 `logs/blogs` 旧 BCFL 运行

除 2026-09-24 的 3 节点、1 轮 smoke test 外，发现 22 个完整 100 轮运行。依据各自 `args_used.txt`，它们都使用：

```text
10 nodes, IID=0, 5 local epochs, threshold=0.1,
noise parameter=2, online rate=1.0,
even link speed=1, even computation power=1, PoS
```

恶意节点数量分组为：

| 恶意节点数 | 完整运行数 |
|---:|---:|
| 1 | 3 |
| 2 | 3 |
| 3 | 4 |
| 4 | 2 |
| 5 | 4 |
| 6 | 2 |

这些运行没有 seed 字段，且当前仓库没有 commit 历史，暂不能正式称为可复核的独立 seed 重复。

### 4.2 图 14 及 BCFL 曲线

`final_plots/figure_3/fig_14.py` 和 `figure_3_bcfl.py`：

- 自动读取 `logs/blogs` 下所有运行；
- 只取 `worker_final_local_accuracies` 中的良性 worker；
- 先对每轮良性 worker 求均值，再对运行求均值；
- 某些运行在后期没有良性 worker 记录，导致每个轮次的实际样本数不同；
- 绘图值是本地更新后的验证 MSE，不是生成样本质量，也不是明确的全局模型指标；
- 图中使用 `MSE`/`ESM Loss` 等名称，不能继续作为生成质量结论。

`final_plots/figure_1/final.py` 直接硬编码 Figure 10 的数值，不能作为由日志可复核生成的图表。

`logs/flogs` 和 `logs/nflogs` 使用不同的旧日志格式；相关脚本读取 `global_comm`、`client_*` 或 `Epoch ..., Test Loss`，不能在没有逐项配置核对的情况下与 `logs/blogs` 混作同一组重复实验。

## 5. 开销日志核查

当前 `accuracy_comm_N.txt` 能记录：

- `comm_round_block_gen_time`；
- `slowest_device_round_ends_time`；
- `comm_round_spent_time_on_this_machine`；
- PoS、分叉和出块节点。

代码内部还计算了本地更新、传输和验证时间，并把部分时间写入 transaction，但没有形成稳定的逐阶段汇总日志。正式计时前应增加统一 schema，并区分实际 wall-clock、带宽公式模拟时间和共识假设时间。

## 6. 阶段 0 放行条件

在开始正式实验前完成以下事项：

1. 建立固定实验配置表和运行元数据 schema，加入 seed、代码版本、数据划分、角色、恶意节点集合和日志字段；
2. 决定标准 DDPM 路线；若不重做 DDPM，则同步收缩论文中的 DDPM、生成质量和 MSE 主张；
3. 实现前向加噪、噪声预测目标和反向采样后，先完成单 batch/小规模 DDPM smoke test；
4. 将训练损失、防御分数和生成质量指标改为不同名称；
5. 在独立先导种子上校准固定近阈值攻击，正式测试前冻结攻击强度；
6. 增加逐验证记录：worker ID、validator ID、真实标签、分数、阈值、过滤决定、最终聚合决定；
7. 先加入计时日志，再启动正式重复实验；
8. 为无区块链但保留同一筛选器的消融预留开关，并明确关闭范围。

在上述条件满足前，不更新论文中的 Figure 10–14 数值、摘要、结论或回复信中的完成时态。
