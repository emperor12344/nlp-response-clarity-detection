# -*- coding: utf-8 -*-
"""第 6 步：改进实验 —— 指令前缀消融。
对应论文的"指令模板"设计，验证加/不加指令前缀对 E5-small 的影响。
三组：A 无前缀（=基线B）/ B query: 前缀 / C 任务指令前缀。
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
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

RNG = 42
random.seed(RNG)
np.random.seed(RNG)
LABELS = ['Clear Reply', 'Ambivalent', 'Clear Non-Reply']
MODEL = 'intfloat/e5-small-v2'

ds = load_dataset('ailsntua/QEvasion')
train = pd.DataFrame(ds['train'])
test = pd.DataFrame(ds['test'])

def base_text(row):
    return "[Q] " + row['question'] + " [A] " + row['interview_answer']

INSTRUCTION = ("Instruct: Given a question and its answer, determine whether the answer "
               "is a clear reply, an ambivalent reply, or a clear non-reply.\nQuery: ")

variants = {
    'A_无前缀(=基线B)': lambda row: base_text(row),
    'B_query前缀':    lambda row: "query: " + base_text(row),
    'C_任务指令前缀': lambda row: INSTRUCTION + base_text(row),
}

y_train = train['clarity_label'].tolist()
y_test = test['clarity_label'].tolist()

print(f"加载句向量模型 {MODEL} ...")
model = SentenceTransformer(MODEL)

results = {}
for name, fn in variants.items():
    tr = train.apply(fn, axis=1).tolist()
    te = test.apply(fn, axis=1).tolist()
    print(f"\n>>> 编码 {name} ...")
    emb_tr = model.encode(tr, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    emb_te = model.encode(te, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RNG)
    clf.fit(emb_tr, y_train)
    pred = clf.predict(emb_te)
    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average='macro')
    results[name] = {'accuracy': round(acc, 4), 'macro_f1': round(mf1, 4)}
    print(f"{name}: Accuracy={acc:.4f}  Macro-F1={mf1:.4f}")

print("\n" + "=" * 50)
print("指令前缀消融结果对比：")
for name, m in results.items():
    print(f"  {name}:  Macro-F1={m['macro_f1']}  Accuracy={m['accuracy']}")

best = max(results.items(), key=lambda kv: kv[1]['macro_f1'])
print(f"\n最佳改进：{best[0]}  (Macro-F1={best[1]['macro_f1']})")

out = {'model': MODEL, 'variants': results, 'best': best[0],
       'note': 'A=无前缀即基线B(0.4715)；B/C为改进。主指标 Macro-F1。'}
with open(BASE + 'results/baselineB_improved_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n已保存：results/baselineB_improved_metrics.json")
