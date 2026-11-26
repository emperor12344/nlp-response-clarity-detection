# -*- coding: utf-8 -*-
"""把 results/*_predictions.csv 转成要求的提交格式：
id, gold, prediction, prob_Clear_Reply, prob_Ambivalent, prob_Clear_Non-Reply
（概率列顺序统一；id 为测试集样本序号，从 0 开始）
"""
import csv
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
import re

RES = BASE + 'results'
OUT_DIR = BASE + 'predictions'
os.makedirs(OUT_DIR, exist_ok=True)

FIELD = ['id', 'gold', 'prediction',
         'prob_Clear_Reply', 'prob_Ambivalent', 'prob_Clear_Non-Reply']
TARGETS = ['Clear Reply', 'Ambivalent', 'Clear Non-Reply']


def norm_key(name):
    return re.sub(r'[\s_\-]', '', name).lower()


def convert(src, dst):
    with open(src, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    prob_col = {}
    for h in header:
        for t in TARGETS:
            if norm_key(h) == 'prob' + norm_key(t):
                prob_col[t] = h
    out_rows = []
    for i, row in enumerate(rows):
        out_rows.append({
            'id': i,
            'gold': row['gold'],
            'prediction': row['prediction'],
            'prob_Clear_Reply': row.get(prob_col.get('Clear Reply', ''), ''),
            'prob_Ambivalent': row.get(prob_col.get('Ambivalent', ''), ''),
            'prob_Clear_Non-Reply': row.get(prob_col.get('Clear Non-Reply', ''), ''),
        })
    with open(dst, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELD)
        w.writeheader()
        w.writerows(out_rows)
    print('written', dst, 'rows', len(out_rows))


for name in ['baselineA_predictions.csv', 'baselineB_predictions.csv', 'finetune_predictions.csv']:
    convert(os.path.join(RES, name), os.path.join(OUT_DIR, name))
