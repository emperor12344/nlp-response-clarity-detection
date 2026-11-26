# -*- coding: utf-8 -*-
"""第 10 步：改进 —— 非线性分类器 MLP（在冻结的 E5-small 句向量上）。

基线 B 是"冻结 E5-small 句向量 + 逻辑回归（线性分类器）"。
这里换成"非线性分类器 MLP（多层感知机）"，看能否学到线性分不开的决策边界。
为公平对比：同样用类别平衡的交叉熵损失、固定随机种子、早停。
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
os.makedirs(BASE + 'results', exist_ok=True)   # 确保输出目录存在（全新环境首次运行）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = BASE + 'data/hf'

import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

RNG = 42
random.seed(RNG)
np.random.seed(RNG)
torch.manual_seed(RNG)

LABELS = ['Clear Reply', 'Ambivalent', 'Clear Non-Reply']
MODEL = 'intfloat/e5-small-v2'
LE = {lab: i for i, lab in enumerate(LABELS)}

ds = load_dataset('ailsntua/QEvasion')
train = pd.DataFrame(ds['train'])
test = pd.DataFrame(ds['test'])


def text(row):
    return "[Q] " + row['question'] + " [A] " + row['interview_answer']


train_texts = train.apply(text, axis=1).tolist()
test_texts = test.apply(text, axis=1).tolist()
y_train = train['clarity_label'].tolist()
y_test = test['clarity_label'].tolist()

# 冻结 E5-small 编码
model = SentenceTransformer(MODEL)
print("编码训练集 ...")
emb_train = model.encode(train_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
print("编码测试集 ...")
emb_test = model.encode(test_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)

np.save(BASE + 'results/e5small_emb_train.npy', emb_train)
np.save(BASE + 'results/e5small_emb_test.npy', emb_test)
print("已保存冻结句向量 .npy，维度", emb_train.shape)

# 类别平衡的标签编码 + 划分
y_train_np = np.array(y_train)
idx = np.arange(len(emb_train))
tr_idx, val_idx = train_test_split(idx, test_size=0.1, stratify=y_train_np, random_state=RNG)

X_tr = torch.tensor(emb_train[tr_idx], dtype=torch.float32)
X_val = torch.tensor(emb_train[val_idx], dtype=torch.float32)
X_test = torch.tensor(emb_test, dtype=torch.float32)
y_tr = torch.tensor([LE[y] for y in y_train_np[tr_idx]], dtype=torch.long)
y_val = torch.tensor([LE[y] for y in y_train_np[val_idx]], dtype=torch.long)
y_test_t = torch.tensor([LE[y] for y in y_test], dtype=torch.long)

cw = compute_class_weight('balanced', classes=np.array(LABELS), y=y_train_np)
weights = torch.tensor(cw, dtype=torch.float32)
print("类别平衡权重：", dict(zip(LABELS, cw.round(3))))


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, n_cls, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, n_cls),
        )

    def forward(self, x):
        return self.net(x)


net = MLP(emb_train.shape[1], 128, len(LABELS))
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=weights)

best_val = -1.0
best_state = None
patience = 15
EPOCHS = 200

print("训练 MLP ...")
for epoch in range(EPOCHS):
    net.train()
    opt.zero_grad()
    loss = criterion(net(X_tr), y_tr)
    loss.backward()
    opt.step()

    net.eval()
    with torch.no_grad():
        val_pred = net(X_val).argmax(1).numpy()
    val_f1 = f1_score(y_val.numpy(), val_pred, average='macro')

    if val_f1 > best_val:
        best_val = val_f1
        best_state = {k: v.clone() for k, v in net.state_dict().items()}
        patience = 15
    else:
        patience -= 1
        if patience == 0:
            print(f"早停于 epoch {epoch + 1}，最佳验证 Macro-F1={best_val:.4f}")
            break

net.load_state_dict(best_state)
net.eval()
with torch.no_grad():
    test_out = net(X_test)
    proba = torch.softmax(test_out, dim=1).numpy()
    test_pred = test_out.argmax(1).numpy()

pred = [LABELS[i] for i in test_pred]

acc = accuracy_score(y_test, pred)
mf1 = f1_score(y_test, pred, average='macro')
p, r, f, _ = precision_recall_fscore_support(y_test, pred, labels=LABELS, zero_division=0)
cm = confusion_matrix(y_test, pred, labels=LABELS)

results = {
    'name': '冻结 E5-small 句向量 + MLP（非线性分类器）',
    'model': MODEL,
    'accuracy': round(acc, 4),
    'macro_precision': round(float(p.mean()), 4),
    'macro_recall': round(float(r.mean()), 4),
    'macro_f1': round(mf1, 4),
    'best_val_macro_f1': round(float(best_val), 4),
    'embedding_dim': int(emb_train.shape[1]),
    'per_class': {LABELS[i]: {'precision': round(float(p[i]), 4),
                              'recall': round(float(r[i]), 4),
                              'f1': round(float(f[i]), 4)} for i in range(len(LABELS))},
    'confusion_matrix': cm.tolist(),
    'labels': LABELS,
    '训练配置': {
        '方法': '冻结句向量 + MLP（128 隐层 + ReLU + Dropout 0.3）',
        'sentence_model': MODEL, 'normalize': True, 'prefix': '无',
        'loss': 'CrossEntropyLoss(weight=balanced)', 'optimizer': 'Adam',
        'lr': 1e-3, 'weight_decay': 1e-4, 'early_stopping': True,
        'val_frac': 0.1, 'random_seed': RNG,
    },
}

with open(BASE + 'results/mlp_metrics.json', 'w', encoding='utf-8') as fh:
    json.dump(results, fh, ensure_ascii=False, indent=2)

out_df = pd.DataFrame({
    'question': test['question'].tolist(),
    'interview_answer': test['interview_answer'].tolist(),
    'gold': y_test,
    'prediction': pred,
})
for i, lab in enumerate(LABELS):
    out_df[f'prob_{lab}'] = proba[:, i]
out_df.to_csv(BASE + 'results/mlp_predictions.csv',
              index=False, encoding='utf-8-sig')

print(f"\nMLP 结果：Accuracy={acc:.4f}  Macro-F1={mf1:.4f}（最佳验证 Macro-F1={best_val:.4f}）")
print("\n" + classification_report(y_test, pred, labels=LABELS, digits=4))
print("混淆矩阵（行=真实，列=预测）：")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
print("\n已保存：results/mlp_metrics.json, results/mlp_predictions.csv")
