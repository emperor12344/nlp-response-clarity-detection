# -*- coding: utf-8 -*-
"""第 8 步：改进 —— TF-IDF 词袋特征 + E5-small 句向量 拼接 + 逻辑回归。
同时捕捉表面词汇特征（如拒答用语）与深层语义特征。
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
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from scipy.sparse import hstack

RNG = 42
random.seed(RNG)
np.random.seed(RNG)
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

# TF-IDF 词袋特征
vec = TfidfVectorizer(ngram_range=(1, 1), max_features=30000, sublinear_tf=True)
X_tr_tfidf = vec.fit_transform(train_texts)
X_te_tfidf = vec.transform(test_texts)

# E5-small 句向量
print(f"加载句向量模型 {MODEL} ...")
model = SentenceTransformer(MODEL)
print("编码训练集 ...")
emb_tr = model.encode(train_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
print("编码测试集 ...")
emb_te = model.encode(test_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)

# 拼接
X_tr = hstack([X_tr_tfidf, emb_tr]).tocsr()
X_te = hstack([X_te_tfidf, emb_te]).tocsr()
print(f"拼接后特征维度：{X_tr.shape[1]}（TF-IDF {X_tr_tfidf.shape[1]} + 句向量 {emb_tr.shape[1]}）")

clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RNG)
clf.fit(X_tr, y_train)
pred = clf.predict(X_te)

acc = accuracy_score(y_test, pred)
mf1 = f1_score(y_test, pred, average='macro')
p, r, f, _ = precision_recall_fscore_support(y_test, pred, labels=LABELS, zero_division=0)
cm = confusion_matrix(y_test, pred, labels=LABELS)

results = {
    'name': 'TF-IDF + E5-small 拼接 + LR',
    'accuracy': round(acc, 4),
    'macro_precision': round(float(p.mean()), 4),
    'macro_recall': round(float(r.mean()), 4),
    'macro_f1': round(mf1, 4),
    'feature_dim': int(X_tr.shape[1]),
    'tfidf_dim': int(X_tr_tfidf.shape[1]),
    'embedding_dim': int(emb_tr.shape[1]),
    'per_class': {LABELS[i]: {'precision': round(float(p[i]), 4),
                              'recall': round(float(r[i]), 4),
                              'f1': round(float(f[i]), 4)} for i in range(len(LABELS))},
    'confusion_matrix': cm.tolist(),
    'labels': LABELS,
    '训练配置': {'tfidf': 'unigram, max_features=30000, sublinear_tf', 'sentence_model': MODEL,
               'classifier': 'LogisticRegression', 'class_weight': 'balanced',
               'max_iter': 2000, 'random_seed': RNG}
}

with open(BASE + 'results/concat_metrics.json', 'w', encoding='utf-8') as fh:
    json.dump(results, fh, ensure_ascii=False, indent=2)

print(f"\n拼接结果：Accuracy={acc:.4f}  Macro-F1={mf1:.4f}")
print("\n" + classification_report(y_test, pred, labels=LABELS, digits=4))
print("混淆矩阵（行=真实，列=预测）：")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
print("\n已保存：results/concat_metrics.json")
