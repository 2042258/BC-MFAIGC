# 阶段 1 状态：DDPM、日志、阈值校准与性能（2026-09-24）

本文档取代同日的 `phase1_ddpm_smoke.md` 初稿，记录当前代码状态、实测数字和待决策项。
所有运行均为本地 pilot/smoke；尚未开始正式实验。

## 1. 代码现状

- `models/Models.py`：标准 DDPM（beta/alpha schedule、`q_sample`、epsilon 预测损失、ancestral 采样）。
- `BFL/DatasetLoad.py`：读取 CIFAR-10 全部 5 个训练 batch（50,000 张）与完整测试集（10,000 张）。
- `BFL/Device.py` /
  - worker 与 validator 使用**相同**的更新协议（相同梯度累积、相同 local epoch 数），消除了原先 validator 每 batch 步进、worker 每 4 batch 步进导致的 4 倍训练量差异；该差异曾使良性分数被训练进度主导（均值约 0.50）。
  - 训练输入转为 `[-1, 1]`；验证在固定 `validation_seed` 下生成噪声，并按元素数加权求均值，评估结果可复现且与 batch size 无关。
  - 每个 (validator, worker) 验证记录写入 `validation_records.jsonl`；最终聚合决定写入 `aggregation_decisions.jsonl`；每轮角色/在线状态写入 `round_metadata.jsonl`（均为合法 JSONL，含真实恶意标签）。
  - 恶意攻击仍为本地训练后对模型权重加 `noise_variance * N(0,1)`，只作用于权重（不含 bias）；噪声尺度参数已改为 float。
- `BFL/main.py`：新增 `--seed`、`--validation_seed`、`--test_batchsize`（默认 256）、`--save_global_checkpoints`、`--checkpoint_frequency`；保存每轮全局模型 checkpoint。
- `BFL/stable_hash.py`：交易/区块哈希改为对张量原始字节做 SHA-256，替代原来的 `str(sorted(...))` 文本哈希（PyTorch 会把大张量截断成省略号，原哈希既不真实又极慢）。
- `BFL/evaluate_ddpm.py`：从 checkpoint 采样并保存 `samples.pt` / `preview.png`。
- `BFL/evaluate_fid.py`：使用官方 `pytorch-fid` InceptionV3（`pt_inception-2015-12-05`，2048-d pre-logits）计算标准 FID；权重文件已下载到 `data/fid_weights/` 并复制到 `~/.cache/torch/hub/checkpoints/`。
- 传输时延模型改用真实 payload 字节数（`payload_size_bytes`），不再使用截断文本的长度。

### 端到端验证

- 3 节点、2 轮、`--checkpoint` 开启、`-cs 1`（开启签名验证）：签名/区块验证全部通过，标签、日志与 checkpoint 正常。
- 2 轮 checkpoint → 采样 256 张 → FID 流程跑通（未训练模型的 FID ≈ 407.7，仅为流程验证数值，不能引用）。

## 2. 性能实测（GTX 1650 Ti，WSL）

| 配置 | 单轮耗时 | 说明 |
|---|---:|---|
| 10 节点、B=128、le=1、旧文本哈希 | 466 s | 优化前 |
| 10 节点、B=128、le=1、稳定哈希 + 评测 batch 256 + 去掉 stake 冗余验证 | **73 s** | 6.4× 提升 |

profile（3 节点）显示：优化前约 36% 时间花在张量转字符串做哈希，24% 花在逐 batch `.to(device)`。

按 73 s/轮估算：10 节点 30 轮 ≈ 40 min/run；50 轮 ≈ 1 h/run；100 轮 ≈ 2 h/run。
正式矩阵（多个恶意比例 × 多个种子 + 消融）需要按此规划 GPU 时间。

## 3. Pilot：良性分数与攻击强度校准

配置：3 节点、1 worker / 1 validator / 1 miner、1 轮、`le=1`、`B=128`、`test_batchsize=256`、非 IID（标签排序分片）、阈值设为 999 仅收集数据。
**所有这些数字都是 pilot，不是论文结果。**

良性验证分数（worker、validator 均为良性；按唯一 seed 去重后约 15 个独立配置）：

```text
范围约 [-0.06, +0.10]，中位数约 -0.01，多数 |score| < 0.1
```

恶意权重噪声尺度 σ 与分数（每个 σ 的独立样本 2–4 个）：

| σ | score 范围 | 结论 |
|---:|---|---|
| 0.005 | -0.019 ~ 0.245 | 大部分与良性重合 |
| 0.01 | 0.026 ~ 0.287 | 与良性上尾重合 |
| 0.015 | 0.099 ~ 0.357 | 接近判定边界 |
| 0.02 | 0.069 ~ 0.411 | 边界附近，部分可检出 |
| 0.05 | 2.62 ~ 7.56 | 极易检出 |

若以良性 99 分位（约 0.10–0.15）设阈值，则**近阈值攻击强度约 σ ≈ 0.01–0.02**；σ ≥ 0.05 属于“易分离攻击”。
该结论验证了“先导校准 → 冻结攻击强度”的方法可行，正式数值需在正式配置下重新校准。

## 4. 待决策项（正式实验前）

1. **正式实验规模**：轮数（30/50/100）、种子数、是否维持 10 节点与 le=1，或调整训练预算；总 GPU 时间需用户确认。
2. **阈值规则**：采用独立 pilot 的良性 99 分位 / `mean + 3σ` / 固定值，需在正式运行前冻结。
3. **攻击比例语义**：恶意节点随机角色导致某些轮没有恶意 worker 提交更新；“10/30/60%”应定义为全体网络比例还是参与 worker 比例，需要固定角色或按轮统计口径说明。
4. **无链消融**：需要在代码中增加“关闭区块链流程但保留同一筛选器”的开关（尚未实现）。
5. **验证成本**：若保持 10 节点 × 100 轮 × 多种子，建议进一步削减冗余评测或使用远端服务器。
6. **DDPM 质量对照**：是否训练一个中央 DDPM（全量数据、同一预算）作为 FID 参照。

## 5. 尚未完成

- 正式 DDPM 训练与 FID 结果；正式阈值校准；近阈值攻击正式实验；链上/链下消融；重复种子统计；异构/在线率实验；分阶段计时报告；论文文字修改。
