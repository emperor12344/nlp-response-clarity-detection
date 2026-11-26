# -*- coding: utf-8 -*-
"""数据效率补充：QNLI 全量训练集（非抽样）上的 TF-IDF 与冻结 E5 结果。

对应作业"完整训练数据"的 100%，作为 step13_data_efficiency.py（训练池 10%/30%/100%）的补充。
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
os.makedirs(BASE + 'results', exist_ok=True)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = BASE + 'data/hf'

import json

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

MODEL = 'intfloat/e5-small-v2'
LE = {0: 'entailment', 1: 'not_entailment'}

ds = load_dataset('nyu-mll/glue', 'qnli')
train = ds['train'].to_pandas()
test = ds['validation'].to_pandas()


def text(row):
    return "[Q] " + str(row['question']) + " [A] " + str(row['sentence'])


tr_texts = train.apply(text, axis=1).tolist()
te_texts = test.apply(text, axis=1).tolist()
y_tr = [LE[v] for v in train['label'].tolist()]
y_te = [LE[v] for v in test['label'].tolist()]
print(f"QNLI 全量训练 {len(tr_texts)} 条，测试 {len(te_texts)} 条")

# TF-IDF + LR
vec = TfidfVectorizer(ngram_range=(1, 1), max_features=30000, sublinear_tf=True)
X_tr = vec.fit_transform(tr_texts)
X_te = vec.transform(te_texts)
clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)
tfidf_f1 = f1_score(y_te, pred, average='macro')
tfidf_acc = accuracy_score(y_te, pred)
print(f"全量 TF-IDF+LR: Acc={tfidf_acc:.4f}, Macro-F1={tfidf_f1:.4f}")

# 冻结 E5 + LR（dev 编码复用缓存 qnli_test_emb.npy）
model = SentenceTransformer(MODEL)
emb_tr = model.encode(tr_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
emb_te = np.load(BASE + 'results/qnli_test_emb.npy')
clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
clf.fit(emb_tr, y_tr)
pred = clf.predict(emb_te)
e5_f1 = f1_score(y_te, pred, average='macro')
e5_acc = accuracy_score(y_te, pred)
print(f"全量 冻结E5+LR: Acc={e5_acc:.4f}, Macro-F1={e5_f1:.4f}")

out = {'训练样本数': len(tr_texts), 'TFIDF_acc': round(tfidf_acc, 4),
       'TFIDF_macro_f1': round(tfidf_f1, 4), 'E5_acc': round(e5_acc, 4),
       'E5_macro_f1': round(e5_f1, 4)}
with open(BASE + 'results/qnli_full_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("已保存 results/qnli_full_metrics.json")
