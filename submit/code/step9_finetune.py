# -*- coding: utf-8 -*-
"""第 9 步：改进 —— 对比微调 E5-small（论文的 InfoNCE / MultipleNegativesRankingLoss）。

思路：冻结模型直接取句向量（基线 B）没有利用 clarity_label 的监督信号。
这里用监督对比学习，把"同标签样本对"拉近、把 batch 内其它样本推远，
让学到的向量对"清晰度三分类"更具区分性，再交给逻辑回归。
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
os.makedirs(BASE + 'results', exist_ok=True)   # 确保输出目录存在（全新环境首次运行）
os.makedirs(BASE + 'models', exist_ok=True)   # 确保输出目录存在（全新环境首次运行）
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

RNG = 42
random.seed(RNG)
np.random.seed(RNG)
torch.manual_seed(RNG)

LABELS = ['Clear Reply', 'Ambivalent', 'Clear Non-Reply']
MODEL = 'intfloat/e5-small-v2'
SAVE_DIR = BASE + 'models/e5-small-finetuned'

ds = load_dataset('ailsntua/QEvasion')
train = pd.DataFrame(ds['train'])
test = pd.DataFrame(ds['test'])


def text(row):
    return "[Q] " + row['question'] + " [A] " + row['interview_answer']


train_texts = train.apply(text, axis=1).tolist()
test_texts = test.apply(text, axis=1).tolist()
y_train = train['clarity_label'].tolist()
y_test = test['clarity_label'].tolist()

# 1. 构造同标签正对 (anchor, positive)
by_label = defaultdict(list)
for i, lab in enumerate(y_train):
    by_label[lab].append(i)

anchors, positives = [], []
rng = np.random.RandomState(RNG)
for i in range(len(train_texts)):
    cands = [j for j in by_label[y_train[i]] if j != i]
    j = int(rng.choice(cands))
    anchors.append(train_texts[i])
    positives.append(train_texts[j])

train_dataset = Dataset.from_dict({'anchor': anchors, 'positive': positives})
print(f"正对数量：{len(anchors)}，类别分布：", {k: len(v) for k, v in by_label.items()})

# 2. 加载模型并微调
model = SentenceTransformer(MODEL)
model.max_seq_length = 512

loss = losses.MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir=BASE + 'models/e5-small-finetuned-ckpt',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=10,
    seed=RNG,
    report_to=[],
    save_strategy='no',
    logging_steps=10,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)

print("开始对比微调 ...")
trainer.train()

model.save_pretrained(SAVE_DIR)
print("已保存微调模型：", SAVE_DIR)

# 3. 用微调后模型重新编码 + 逻辑回归
print("微调后重新编码训练集 ...")
emb_train = model.encode(train_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
print("微调后重新编码测试集 ...")
emb_test = model.encode(test_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)

clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RNG)
clf.fit(emb_train, y_train)
pred = clf.predict(emb_test)
proba = clf.predict_proba(emb_test)

acc = accuracy_score(y_test, pred)
mf1 = f1_score(y_test, pred, average='macro')
p, r, f, _ = precision_recall_fscore_support(y_test, pred, labels=LABELS, zero_division=0)
cm = confusion_matrix(y_test, pred, labels=LABELS)

results = {
    'name': '对比微调 E5-small + LR',
    'model': MODEL,
    'accuracy': round(acc, 4),
    'macro_precision': round(float(p.mean()), 4),
    'macro_recall': round(float(r.mean()), 4),
    'macro_f1': round(mf1, 4),
    'embedding_dim': int(emb_train.shape[1]),
    'per_class': {LABELS[i]: {'precision': round(float(p[i]), 4),
                              'recall': round(float(r[i]), 4),
                              'f1': round(float(f[i]), 4)} for i in range(len(LABELS))},
    'confusion_matrix': cm.tolist(),
    'labels': LABELS,
    '训练配置': {
        '方法': '监督对比微调（同标签正对 + MultipleNegativesRankingLoss/InfoNCE）',
        'sentence_model': MODEL, 'batch_size': 16, 'epochs': 1, 'lr': 2e-5,
        'warmup_steps': 10, 'max_seq_length': 512, 'normalize': True, 'prefix': '无',
        'classifier': 'LogisticRegression', 'class_weight': 'balanced',
        'max_iter': 2000, 'random_seed': RNG,
    },
}

with open(BASE + 'results/finetune_metrics.json', 'w', encoding='utf-8') as fh:
    json.dump(results, fh, ensure_ascii=False, indent=2)

out_df = pd.DataFrame({
    'question': test['question'].tolist(),
    'interview_answer': test['interview_answer'].tolist(),
    'gold': y_test,
    'prediction': pred.tolist(),
})
for i, lab in enumerate(clf.classes_):
    out_df[f'prob_{lab}'] = proba[:, i]
out_df.to_csv(BASE + 'results/finetune_predictions.csv',
              index=False, encoding='utf-8-sig')

print(f"\n对比微调结果：Accuracy={acc:.4f}  Macro-F1={mf1:.4f}")
print("\n" + classification_report(y_test, pred, labels=LABELS, digits=4))
print("混淆矩阵（行=真实，列=预测）：")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
print("\n已保存：results/finetune_metrics.json, results/finetune_predictions.csv")
