# 对话问答回避检测与跨场景迁移分析 —— 提交说明

任务：在 QEvasion 数据集上完成"问答回避检测"三分类，并把同一套
"文本对分类"流程迁移到 QNLI（是否包含答案）上验证可复用性。


## 一、环境与依赖

- 操作系统：Windows 11
- Python：3.14.2
- 依赖库：
  - `datasets`（HuggingFace 数据集加载）
  - `pandas`、`numpy`
  - `scikit-learn`（TF-IDF、LogisticRegression、指标）
  - `sentence-transformers` 6.2.0（E5-small 句向量、对比微调）
  - `torch`（sentence-transformers 底层）
  - `matplotlib`、`seaborn`（画图）

国内访问 HuggingFace 需走镜像（脚本开头已设置）：

```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = 'data/hf'   # 模型与数据缓存目录
```

> 所有脚本用脚本自身位置定位：`BASE = code/ 的上级目录`，`data/`、`results/`、`models/`、`predictions/` 都挂在 `BASE` 下，因此整个提交目录拷到新机器即可复现。

## 二、随机种子与关键配置

- **随机种子统一为 42**（`RNG = 42`，覆盖 sklearn 的 `random_state`，`step12` 额外固定 `random/numpy`）。
- 逻辑回归统一 `max_iter=2000, class_weight='balanced'`。
- TF-IDF：`ngram_range=(1,1), max_features=30000, sublinear_tf=True`。
- 句向量：`intfloat/e5-small-v2`（384 维），输入格式 `"[Q] {question} [A] {interview_answer}"`。
- 主指标 **Macro-F1**，辅助 Accuracy、每类 Precision/Recall/F1、混淆矩阵。

## 三、快速复现

在 `code/` 目录下：

```bash
# 一键跑核心必做链路（数据→EDA→画图→基线A/B→概率→误判分析），约几分钟
python run_all.py

# 额外跑迁移/微调/消融/数据效率等实验（较慢，微调在 CPU 上约 1~2 小时）
python run_all.py --full
```

## 四、分步运行说明

| 脚本 | 作用 | 产出 |
|---|---|---|
| `step1_download.py` | 下载 QEvasion 数据集并打印结构 | 数据缓存 `data/hf/` |
| `step2_eda.py` | 探索性数据分析（标签/字段/缺失） | `results/eda_summary.txt` |
| `step3_charts.py` | 画图1标签分布、图2长度分布 | `results/fig1_label_dist.png`、`fig2_length_dist.png` |
| `step4_baselineA.py` | 基线A：TF-IDF + 逻辑回归 | `results/baselineA_metrics.json` |
| `step5_baselineB.py` | 基线B：冻结 E5-small + 逻辑回归 | `results/baselineB_metrics.json` |
| `step10_mlp.py` | MLP 分类器，并保存 E5-small 句向量 | `results/e5small_emb_*.npy` |
| `step12_proba.py` | 给预测补三类概率列（复用句向量） | `results/baselineA/B_predictions.csv` |
| `step14_find_misclass.py` | 挑典型误判样本 | 误判候选 |
| `step9_finetune.py` | 对比微调 E5-small（InfoNCE，慢） | `results/finetune_*.json` |
| `step11_qnli.py` | QNLI 跨场景迁移（慢），并落盘微调模型预测 | `results/qnli_metrics.json`、`predictions/qnli_predictions.csv` |
| `step13_data_efficiency.py` | 数据效率 10%/30%/100% | `results/data_efficiency.json` |
| `step16_ablation.py` | 指令前缀 [Q]/[A] 消融 | `results/ablation_*.json` |
| `make_predictions.py` | 把预测文件转成提交格式（含 id） | `predictions/*.csv` |
| `make_qnli_predictions.py` | 从已保存微调模型生成 QNLI 预测（跳过微调） | `predictions/qnli_predictions.csv` |

> 注：`step12` 复用 `step10` 保存的句向量 `e5small_emb_*.npy`，因此 `step10` 必须先于 `step12` 运行。

其余随附脚本：`step6_improve.py`（前缀消融早期版，与 `step16` 重复）、`step7_e5base.py`（E5-base 骨干）、`step8_concat.py`（TF-IDF 与句向量拼接）为改进探索；`step15_stats.py` 补表1的数据统计；`step_error_analysis.py` 输出误判样本分组统计（`results/error_candidates.txt`）。

## 五、预测文件格式

`predictions/` 下共四个文件，每行一个测试样本。

QEvasion（三分类）三个文件，格式：

```
id, gold, prediction, prob_Clear_Reply, prob_Ambivalent, prob_Clear_Non-Reply
```

QNLI（二分类）一个文件，格式：

```
id, gold, prediction, prob_entailment, prob_not_entailment
```

- `id`：测试集样本序号（0 开始）。
- `gold`：真实标签；`prediction`：模型预测标签。
- `prob_*`：各类别的预测概率（和为 1）。

| 文件 | 任务 | 模型 | Macro-F1 |
|---|---|---|---|
| `baselineA_predictions.csv` | QEvasion | TF-IDF + LR（必做基线） | 0.4147 |
| `baselineB_predictions.csv` | QEvasion | 冻结 E5-small + LR（最佳） | 0.4715 |
| `finetune_predictions.csv` | QEvasion | 对比微调 E5-small + LR | 0.4360 |
| `qnli_predictions.csv` | QNLI | 对比微调 E5-small + LR | 0.8305 |

## 六、迁移配置说明

`migration_config.json` 把 QEvasion 与 QNLI 统一成 `text_a / text_b / label / group` 四个字段：

- QEvasion：`text_a=question`、`text_b=interview_answer`、`label=clarity_label`（三分类）、`group=president`。
- QNLI：`text_a=question`、`text_b=sentence`、`label=label`（0=entailment / 1=not_entailment 二分类）。

训练、评估与可视化代码保持不变，只改字段映射与标签映射，即可从原任务切到迁移任务。
