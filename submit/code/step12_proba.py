# -*- coding: utf-8 -*-
"""第 12 步：给 baselineA / baselineB 的预测补上「每个样本各类别概率」列。

报告要求每个误判样本给出各类别概率，但之前只存了 gold / prediction。
- baselineA：重跑 TF-IDF(unigram) + LR，输出 predict_proba
- baselineB：复用已存好的冻结 E5-small 句向量 (e5small_emb_*.npy) + LR，
            输出 predict_proba，并校验复现的 Macro-F1 是否仍等于 0.4715
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
os.makedirs(BASE + 'results', exist_ok=True)   # 确保输出目录存在（全新环境首次运行）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = BASE + 'data/hf'

import random

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

RNG = 42
random.seed(RNG)
np.random.seed(RNG)

LABELS = ['Clear Reply', 'Ambivalent', 'Clear Non-Reply']
PROB_COLS = [f'prob_{l.replace(" ", "_")}' for l in LABELS]

ds = load_dataset('ailsntua/QEvasion')
train = pd.DataFrame(ds['train'])
test = pd.DataFrame(ds['test'])


def text(row):
    return "[Q] " + row['question'] + " [A] " + row['interview_answer']


def save_proba(out_df, path):
    out_df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  已保存 {path}（{len(out_df)} 行，含概率列 {PROB_COLS}）")


# baselineA：TF-IDF + LR
print("baselineA：TF-IDF + LR ...")
vec = TfidfVectorizer(ngram_range=(1, 1), max_features=30000, sublinear_tf=True)
X_tr = vec.fit_transform(train.apply(text, axis=1).tolist())
X_te = vec.transform(test.apply(text, axis=1).tolist())
y_tr = train['clarity_label'].tolist()
y_te = test['clarity_label'].tolist()

clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RNG)
clf.fit(X_tr, y_tr)
proba = clf.predict_proba(X_te)
pred = clf.predict(X_te)
print(f"  Accuracy={accuracy_score(y_te, pred):.4f}  Macro-F1={f1_score(y_te, pred, average='macro'):.4f}")

dfA = test[['question', 'interview_answer']].copy()
dfA['gold'] = y_te
dfA['prediction'] = pred
for k, c in enumerate(clf.classes_):
    dfA[f'prob_{c.replace(" ", "_")}'] = proba[:, k].round(4)
save_proba(dfA, BASE + 'results/baselineA_predictions.csv')

# baselineB：复用冻结句向量 + LR
print("baselineB：复用 e5small_emb_*.npy + LR ...")
emb_tr = np.load(BASE + 'results/e5small_emb_train.npy')
emb_te = np.load(BASE + 'results/e5small_emb_test.npy')

clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RNG)
clf.fit(emb_tr, y_tr)
proba = clf.predict_proba(emb_te)
pred = clf.predict(emb_te)
acc = accuracy_score(y_te, pred)
mf1 = f1_score(y_te, pred, average='macro')
print(f"  Accuracy={acc:.4f}  Macro-F1={mf1:.4f}（报告值为 Acc=0.5682 / Macro-F1=0.4715）")
ok = abs(mf1 - 0.4715) < 1e-4 and abs(acc - 0.5682) < 1e-4
print("  校验：", "一致 [OK]" if ok else "不一致，需重跑编码")

dfB = test[['question', 'interview_answer']].copy()
dfB['gold'] = y_te
dfB['prediction'] = pred
for k, c in enumerate(clf.classes_):
    dfB[f'prob_{c.replace(" ", "_")}'] = proba[:, k].round(4)
save_proba(dfB, BASE + 'results/baselineB_predictions.csv')

print("\n完成。")
