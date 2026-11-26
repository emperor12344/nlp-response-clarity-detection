# -*- coding: utf-8 -*-
"""第 5 步：基线B —— 句向量（E5-small）+ 逻辑回归。
对应论文 Wang et al. (ACL 2024) 的句向量方法，骨干用 E5-small。
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
os.makedirs(BASE + 'results', exist_ok=True)   # 确保输出目录存在（全新环境首次运行）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = BASE + 'data/hf'

import json
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

RNG = 42
LABELS = ['Clear Reply', 'Ambivalent', 'Clear Non-Reply']
MODEL = 'intfloat/e5-small-v2'

ds = load_dataset('ailsntua/QEvasion')
train = pd.DataFrame(ds['train'])
test = pd.DataFrame(ds['test'])

def text(row):
    return "[Q] " + row['question'] + " [A] " + row['interview_answer']

train_texts = train.apply(text, axis=1).tolist()
test_texts = test.apply(text, axis=1).tolist()
y_train = train['clarity_label'].tolist()
y_test = test['clarity_label'].tolist()

print(f"加载句向量模型 {MODEL} ...")
model = SentenceTransformer(MODEL)

print("编码训练集 ...")
emb_train = model.encode(train_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
print("编码测试集 ...")
emb_test = model.encode(test_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)

clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RNG)
clf.fit(emb_train, y_train)
pred = clf.predict(emb_test)

acc = accuracy_score(y_test, pred)
mf1 = f1_score(y_test, pred, average='macro')
p, r, f, _ = precision_recall_fscore_support(y_test, pred, labels=LABELS, zero_division=0)
cm = confusion_matrix(y_test, pred, labels=LABELS)

results = {
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
    '训练配置': {'sentence_model': MODEL, 'normalize': True,
               'classifier': 'LogisticRegression', 'class_weight': 'balanced',
               'max_iter': 2000, 'random_seed': RNG}
}

with open(BASE + 'results/baselineB_metrics.json', 'w', encoding='utf-8') as fh:
    json.dump(results, fh, ensure_ascii=False, indent=2)

pd.DataFrame({'question': test['question'].tolist(),
              'interview_answer': test['interview_answer'].tolist(),
              'gold': y_test, 'prediction': pred.tolist()}).to_csv(
    BASE + 'results/baselineB_predictions.csv', index=False, encoding='utf-8-sig')

print(f"\n基线B结果：Accuracy={acc:.4f}  Macro-F1={mf1:.4f}  (嵌入维度={emb_train.shape[1]})")
print("\n" + classification_report(y_test, pred, labels=LABELS, digits=4))
print("混淆矩阵（行=真实，列=预测）：")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
print("\n已保存：results/baselineB_metrics.json, results/baselineB_predictions.csv")
