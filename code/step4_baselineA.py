# -*- coding: utf-8 -*-
"""第 4 步：基线A —— TF-IDF + 逻辑回归。
会对比 unigram 与 unigram+bigram，记录多数类基线，输出指标和预测文件。
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
os.makedirs(BASE + 'results', exist_ok=True)   # 确保输出目录存在（全新环境首次运行）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = BASE + 'data/hf'

import json
from collections import Counter
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

RNG = 42
LABELS = ['Clear Reply', 'Ambivalent', 'Clear Non-Reply']

ds = load_dataset('ailsntua/QEvasion')
train = pd.DataFrame(ds['train'])
test = pd.DataFrame(ds['test'])

def text(row):
    return "[Q] " + row['question'] + " [A] " + row['interview_answer']

X_train_raw = train.apply(text, axis=1).tolist()
X_test_raw = test.apply(text, axis=1).tolist()
y_train = train['clarity_label'].tolist()
y_test = test['clarity_label'].tolist()

results = {}

# 多数类基线
maj = Counter(y_train).most_common(1)[0][0]
maj_pred = [maj] * len(y_test)
results['多数类基线'] = {
    'accuracy': round(accuracy_score(y_test, maj_pred), 4),
    'macro_f1': round(f1_score(y_test, maj_pred, average='macro'), 4),
    'note': f'全部预测为 {maj}'
}

# TF-IDF + 逻辑回归：对比 unigram vs unigram+bigram
best = None
for name, ng in [('TF-IDF(unigram)+LR', (1, 1)), ('TF-IDF(uni+bigram)+LR', (1, 2))]:
    vec = TfidfVectorizer(ngram_range=ng, max_features=30000, sublinear_tf=True)
    X_tr = vec.fit_transform(X_train_raw)
    X_te = vec.transform(X_test_raw)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RNG)
    clf.fit(X_tr, y_train)
    pred = clf.predict(X_te)
    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average='macro')
    results[name] = {'accuracy': round(acc, 4), 'macro_f1': round(mf1, 4),
                     'n_features': int(X_tr.shape[1])}
    print(f"{name}:  Accuracy={acc:.4f}  Macro-F1={mf1:.4f}  (特征数={X_tr.shape[1]})")
    if best is None or mf1 > best[1]:
        best = (name, mf1, clf, pred, vec)

best_name, best_mf1, clf, pred, vec = best
print(f"\n最佳配置：{best_name}")

# 最佳配置的详细结果
p, r, f, _ = precision_recall_fscore_support(y_test, pred, labels=LABELS, zero_division=0)
per_class = {}
for i, lab in enumerate(LABELS):
    per_class[lab] = {'precision': round(float(p[i]), 4),
                      'recall': round(float(r[i]), 4),
                      'f1': round(float(f[i]), 4)}
cm = confusion_matrix(y_test, pred, labels=LABELS)

results['最佳模型'] = {
    'name': best_name,
    'accuracy': round(accuracy_score(y_test, pred), 4),
    'macro_precision': round(float(p.mean()), 4),
    'macro_recall': round(float(r.mean()), 4),
    'macro_f1': round(f1_score(y_test, pred, average='macro'), 4),
    'per_class': per_class,
    'confusion_matrix': cm.tolist(),
    'labels': LABELS,
    '训练配置': {'ngram': '1-2' if 'bigram' in best_name else '1-1',
               'max_features': 30000, 'sublinear_tf': True,
               'class_weight': 'balanced', 'max_iter': 2000, 'random_seed': RNG}
}

# 保存
with open(BASE + 'results/baselineA_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

pred_df = pd.DataFrame({'question': test['question'].tolist(),
                        'interview_answer': test['interview_answer'].tolist(),
                        'gold': y_test, 'prediction': pred.tolist()})
pred_df.to_csv(BASE + 'results/baselineA_predictions.csv', index=False, encoding='utf-8-sig')

print("\n" + classification_report(y_test, pred, labels=LABELS, digits=4))
print("混淆矩阵（行=真实，列=预测）：")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
print("\n已保存：results/baselineA_metrics.json, results/baselineA_predictions.csv")
