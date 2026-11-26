# -*- coding: utf-8 -*-
"""第 11 步：跨场景迁移 —— QNLI（question + sentence → 是否含答案，二分类）。

用统一的字段适配器 (text_a=question, text_b=sentence, label) 复用 QEvasion 的
"编码 -> 分类"流程，验证流程的可复用性。四个设置：
  1) 原方法基线     ：QEvasion 上 E5-small+LR 的结果（引用 0.4715，不重算）
  2) 迁移数据基线   ：QNLI 上 TF-IDF + LR
  3) 直接迁移       ：QNLI 上冻结 E5-small + LR（预训练表示直接复用）
  4) 适配后训练     ：在 QNLI 上对比微调 E5-small + LR（模型参数适配新任务）
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
os.makedirs(BASE + 'results', exist_ok=True)   # 确保输出目录存在（全新环境首次运行）
os.makedirs(BASE + 'models', exist_ok=True)   # 确保输出目录存在（全新环境首次运行）
os.makedirs(BASE + 'predictions', exist_ok=True)   # 确保输出目录存在（全新环境首次运行）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = BASE + 'data/hf'

import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                  SentenceTransformerTrainingArguments, losses)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split

RNG = 42
random.seed(RNG)
np.random.seed(RNG)
torch.manual_seed(RNG)

MODEL = 'intfloat/e5-small-v2'
N_TRAIN = 10000          # 从 QNLI train 里抽样用于训练（保证微调够快）
MAX_SEQ = 128            # QNLI 文本短，迁移时把最大长度从 512 降到 128
LABELS = ['entailment', 'not_entailment']   # 0=含答案, 1=不含答案
LE = {0: 'entailment', 1: 'not_entailment'}

# 1. 读 QNLI
ds = load_dataset('nyu-mll/glue', 'qnli')
train = ds['train'].to_pandas()
test = ds['validation'].to_pandas()   # 用官方验证集当测试（有标签）

# 抽样 1 万条（分层，保证两类均衡）
idx = np.arange(len(train))
tr_idx, _ = train_test_split(idx, train_size=N_TRAIN, stratify=train['label'].values,
                             random_state=RNG)
sub = train.iloc[tr_idx].reset_index(drop=True)

def text(row):
    return "[Q] " + str(row['question']) + " [A] " + str(row['sentence'])

tr_texts = sub.apply(text, axis=1).tolist()
te_texts = test.apply(text, axis=1).tolist()
y_tr = [LE[v] for v in sub['label'].tolist()]
y_te = [LE[v] for v in test['label'].tolist()]

print(f"QNLI 训练抽样 {len(tr_texts)}，测试 {len(te_texts)}")
print("训练标签分布:", pd.Series(y_tr).value_counts().to_dict())

results = {}
results['1_原方法基线'] = {'name': 'QEvasion E5-small+LR（原任务）', 'accuracy': 0.5682, 'macro_f1': 0.4715}

# 2. 迁移数据基线：TF-IDF + LR
vec = TfidfVectorizer(ngram_range=(1, 1), max_features=30000, sublinear_tf=True)
X_tr = vec.fit_transform(tr_texts)
X_te = vec.transform(te_texts)
clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RNG)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)
results['2_迁移数据基线'] = {
    'name': 'QNLI TF-IDF+LR',
    'accuracy': round(accuracy_score(y_te, pred), 4),
    'macro_f1': round(f1_score(y_te, pred, average='macro'), 4),
}

# 3. 直接迁移：冻结 E5-small + LR
model = SentenceTransformer(MODEL)
print("直接迁移：编码 QNLI ...")
emb_tr = model.encode(tr_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
emb_te = model.encode(te_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RNG)
clf.fit(emb_tr, y_tr)
pred = clf.predict(emb_te)
results['3_直接迁移'] = {
    'name': 'QNLI 冻结 E5-small+LR',
    'accuracy': round(accuracy_score(y_te, pred), 4),
    'macro_f1': round(f1_score(y_te, pred, average='macro'), 4),
}

# 4. 适配后训练：对比微调 E5-small + LR
by_label = defaultdict(list)
for i, lab in enumerate(y_tr):
    by_label[lab].append(i)
anchors, positives = [], []
rng = np.random.RandomState(RNG)
for i in range(len(tr_texts)):
    cands = [j for j in by_label[y_tr[i]] if j != i]
    j = int(rng.choice(cands))
    anchors.append(tr_texts[i])
    positives.append(tr_texts[j])

ft_model = SentenceTransformer(MODEL)
ft_model.max_seq_length = MAX_SEQ
loss = losses.MultipleNegativesRankingLoss(ft_model)
train_dataset = Dataset.from_dict({'anchor': anchors, 'positive': positives})
args = SentenceTransformerTrainingArguments(
    output_dir=BASE + 'models/e5-small-qnli-ckpt',
    num_train_epochs=1,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=10,
    seed=RNG,
    report_to=[],
    save_strategy='no',
)
trainer = SentenceTransformerTrainer(model=ft_model, args=args,
                                     train_dataset=train_dataset, loss=loss)
print("适配后训练：QNLI 对比微调 ...")
trainer.train()
ft_model.save_pretrained(BASE + 'models/e5-small-qnli')

print("适配后：重新编码 QNLI ...")
emb_tr = ft_model.encode(tr_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
emb_te = ft_model.encode(te_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RNG)
clf.fit(emb_tr, y_tr)
pred = clf.predict(emb_te)
results['4_适配后训练'] = {
    'name': 'QNLI 对比微调 E5-small+LR',
    'accuracy': round(accuracy_score(y_te, pred), 4),
    'macro_f1': round(f1_score(y_te, pred, average='macro'), 4),
}

# 落盘最佳模型（对比微调）的预测文件
_cls = clf.classes_.tolist()
_proba = clf.predict_proba(emb_te)
qnli_df = pd.DataFrame({
    'id': range(len(te_texts)),
    'gold': y_te,
    'prediction': pred.tolist(),
    'prob_entailment': _proba[:, _cls.index('entailment')],
    'prob_not_entailment': _proba[:, _cls.index('not_entailment')],
})
qnli_out = BASE + 'predictions/qnli_predictions.csv'
os.makedirs(os.path.dirname(qnli_out), exist_ok=True)
qnli_df.to_csv(qnli_out, index=False, encoding='utf-8-sig')
print('已保存 predictions/qnli_predictions.csv（', len(qnli_df), '行）')

results['配置'] = {
    '迁移数据': 'QNLI (nyu-mll/glue)', '训练抽样': N_TRAIN, '测试': len(te_texts),
    '字段适配器': 'text_a=question, text_b=sentence, label=0/1',
    'max_seq_length': MAX_SEQ, 'random_seed': RNG,
}

with open(BASE + 'results/qnli_metrics.json', 'w', encoding='utf-8') as fh:
    json.dump(results, fh, ensure_ascii=False, indent=2)

print("\nQNLI 迁移结果")
for k in ['1_原方法基线', '2_迁移数据基线', '3_直接迁移', '4_适配后训练']:
    r = results[k]
    print(f"{r['name']:32s}  Acc={r['accuracy']}  Macro-F1={r['macro_f1']}")
print("已保存 results/qnli_metrics.json")
