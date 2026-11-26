# -*- coding: utf-8 -*-
"""第 13 步：数据效率扩展（大作业的扩展点）。

在 QNLI 上，分别用 10% / 30% / 100%（1000 / 3000 / 10000 条）训练，
对比「TF-IDF + LR」和「冻结 E5-small + LR」两条曲线，看预训练表示是否在
少数据时更有优势（few-shot / data efficiency）。

复用 step11 的字段适配器与统一流程，只改训练数据量。
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
os.makedirs(BASE + 'results', exist_ok=True)   # 确保输出目录存在（全新环境首次运行）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = BASE + 'data/hf'

import json
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

RNG = 42
random.seed(RNG)
np.random.seed(RNG)

MODEL = 'intfloat/e5-small-v2'
POOL = 10000                    # 100% = 从 QNLI train 抽 1 万条
FRACTIONS = [0.1, 0.3, 1.0]     # 10% / 30% / 100%
SIZES = [int(POOL * f) for f in FRACTIONS]
LE = {0: 'entailment', 1: 'not_entailment'}

# 中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
BLUE, ORANGE = '#2a78d6', '#eb6834'
SURFACE, INK, MUTED, GRID = '#ffffff', '#0b0b0b', '#898781', '#e1e0d9'

# ---- 读 QNLI ----
ds = load_dataset('nyu-mll/glue', 'qnli')
train = ds['train'].to_pandas()
test = ds['validation'].to_pandas()

idx = np.arange(len(train))
pool_idx, _ = train_test_split(idx, train_size=POOL, stratify=train['label'].values,
                               random_state=RNG)
pool = train.iloc[pool_idx].reset_index(drop=True)


def text(row):
    return "[Q] " + str(row['question']) + " [A] " + str(row['sentence'])


pool_texts = pool.apply(text, axis=1).tolist()
pool_y = [LE[v] for v in pool['label'].tolist()]
te_texts = test.apply(text, axis=1).tolist()
te_y = [LE[v] for v in test['label'].tolist()]

print(f"QNLI 训练池 {POOL} 条，测试 {len(te_y)} 条")

# 冻结 E5：把整个训练池编码一次，之后各比例只切片（不用重复编码）
pool_emb_path = BASE + 'results/qnli_pool_emb.npy'
test_emb_path = BASE + 'results/qnli_test_emb.npy'
if os.path.exists(pool_emb_path) and os.path.exists(test_emb_path):
    print("复用已编码的句向量 .npy ...")
    pool_emb = np.load(pool_emb_path)
    test_emb = np.load(test_emb_path)
else:
    model = SentenceTransformer(MODEL)
    print("编码训练池 + 测试集（E5 冻结）...")
    pool_emb = model.encode(pool_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    test_emb = model.encode(te_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    np.save(pool_emb_path, pool_emb)
    np.save(test_emb_path, test_emb)

rows = []
for frac, size in zip(FRACTIONS, SIZES):
    if size >= POOL:
        sub_idx = np.arange(POOL)
    else:
        sub_idx, _ = train_test_split(np.arange(POOL), train_size=size,
                                      stratify=np.array(pool_y), random_state=RNG)
        sub_idx = np.sort(sub_idx)
    sub_texts = [pool_texts[i] for i in sub_idx]
    sub_y = [pool_y[i] for i in sub_idx]

    # TF-IDF + LR
    vec = TfidfVectorizer(ngram_range=(1, 1), max_features=30000, sublinear_tf=True)
    X = vec.fit_transform(sub_texts)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RNG)
    clf.fit(X, sub_y)
    tfidf_f1 = round(f1_score(te_y, clf.predict(vec.transform(te_texts)), average='macro'), 4)

    # 冻结 E5 + LR（复用已编码的池向量，只切片）
    X_emb = pool_emb[sub_idx]
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RNG)
    clf.fit(X_emb, sub_y)
    e5_f1 = round(f1_score(te_y, clf.predict(test_emb), average='macro'), 4)

    rows.append({'比例': f'{int(frac*100)}%', '训练样本数': size,
                 'TFIDF_Macro_F1': tfidf_f1, 'E5_Macro_F1': e5_f1})
    print(f"{int(frac*100):3d}% ({size:5d} 条):  TF-IDF F1={tfidf_f1}   E5 F1={e5_f1}")

df = pd.DataFrame(rows)
out = {'任务': '数据效率扩展（QNLI）', '训练池': POOL, 'random_seed': RNG,
       'rows': rows,
       'gap': {r['比例']: round(r['E5_Macro_F1'] - r['TFIDF_Macro_F1'], 4) for r in rows}}
with open(BASE + 'results/data_efficiency.json', 'w', encoding='utf-8') as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2)

# ---- 学习曲线 ----
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(SIZES, [r['TFIDF_Macro_F1'] for r in rows], 'o-', color=ORANGE, lw=2,
        label='TF-IDF + LR', markersize=7)
ax.plot(SIZES, [r['E5_Macro_F1'] for r in rows], 's-', color=BLUE, lw=2,
        label='冻结 E5-small + LR', markersize=7)
for x, r in zip(SIZES, rows):
    ax.annotate(f"{r['TFIDF_Macro_F1']}", (x, r['TFIDF_Macro_F1']), textcoords='offset points',
                xytext=(0, -16), ha='center', fontsize=9, color=MUTED)
    ax.annotate(f"{r['E5_Macro_F1']}", (x, r['E5_Macro_F1']), textcoords='offset points',
                xytext=(0, 9), ha='center', fontsize=9, color=INK)
ax.set_xlabel('训练样本数（QNLI）', fontsize=11, color=MUTED)
ax.set_ylabel('Macro-F1', fontsize=11, color=MUTED)
ax.set_title('数据效率对比：预训练表示 vs 词袋', fontsize=13, color=INK, pad=12)
ax.set_xscale('log')
ax.set_xticks(SIZES)
ax.set_xticklabels([str(s) for s in SIZES])
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED)
fig.tight_layout()
fig.savefig(BASE + 'results/fig_data_efficiency.png', dpi=150, facecolor=SURFACE)
plt.close(fig)

print("\n学习曲线已保存：results/fig_data_efficiency.png")
print("结果已保存：results/data_efficiency.json")
print("\n各比例下 E5 相对 TF-IDF 的领先幅度：")
for r in rows:
    print(f"  {r['比例']}: +{r['E5_Macro_F1'] - r['TFIDF_Macro_F1']:.4f}")
