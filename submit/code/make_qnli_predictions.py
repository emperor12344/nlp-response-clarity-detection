# -*- coding: utf-8 -*-
"""从已保存的 QNLI 微调模型 models/e5-small-qnli 生成预测文件，无需重跑微调。

前提：已先跑过 step11_qnli.py（得到 models/e5-small-qnli）。
输出：predictions/qnli_predictions.csv
      （id, gold, prediction, prob_entailment, prob_not_entailment）
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = BASE + 'data/hf'

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

RNG = 42
MODEL_DIR = BASE + 'models/e5-small-qnli'
N_TRAIN = 10000
LE = {0: 'entailment', 1: 'not_entailment'}

ds = load_dataset('nyu-mll/glue', 'qnli')
train = ds['train'].to_pandas()
test = ds['validation'].to_pandas()

idx = np.arange(len(train))
tr_idx, _ = train_test_split(idx, train_size=N_TRAIN, stratify=train['label'].values, random_state=RNG)
sub = train.iloc[tr_idx].reset_index(drop=True)


def text(row):
    return "[Q] " + str(row['question']) + " [A] " + str(row['sentence'])


tr_texts = sub.apply(text, axis=1).tolist()
te_texts = test.apply(text, axis=1).tolist()
y_tr = [LE[v] for v in sub['label'].tolist()]
y_te = [LE[v] for v in test['label'].tolist()]

model = SentenceTransformer(MODEL_DIR)
print('编码训练集 ...')
emb_tr = model.encode(tr_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
print('编码测试集 ...')
emb_te = model.encode(te_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)

clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RNG)
clf.fit(emb_tr, y_tr)
pred = clf.predict(emb_te)
proba = clf.predict_proba(emb_te)
cls = clf.classes_.tolist()

df = pd.DataFrame({
    'id': range(len(te_texts)),
    'gold': y_te,
    'prediction': pred.tolist(),
    'prob_entailment': proba[:, cls.index('entailment')],
    'prob_not_entailment': proba[:, cls.index('not_entailment')],
})
out = BASE + 'predictions/qnli_predictions.csv'
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False, encoding='utf-8-sig')

print('已保存', out)
print('Macro-F1 =', round(f1_score(y_te, pred, average='macro'), 4))
