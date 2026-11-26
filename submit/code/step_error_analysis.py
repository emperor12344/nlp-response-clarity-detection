# -*- coding: utf-8 -*-
"""从 baselineB_predictions.csv 导出误判样本候选 + 长度统计，供补表5与误差-长度分析。"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
os.makedirs(BASE + 'results', exist_ok=True)   # 确保输出目录存在（全新环境首次运行）
import csv
from collections import defaultdict

CSV = BASE + 'results/baselineB_predictions.csv'
OUT = BASE + 'results/error_candidates.txt'

rows = []
with open(CSV, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        rows.append(r)

mis = [r for r in rows if r['gold'] != r['prediction']]
ok = [r for r in rows if r['gold'] == r['prediction']]

groups = defaultdict(list)
for r in mis:
    groups[(r['gold'], r['prediction'])].append(r)

def wc(s):
    return len(s.split())

lines = []
lines.append(f'总测试样本 308，误判 {len(mis)}，正确 {len(ok)}')
lines.append('')
# 长度统计
def avg(items):
    return sum(wc(r['interview_answer']) for r in items) / len(items) if items else 0
lines.append('回答长度（词数）：正确样本平均 %.1f，误判样本平均 %.1f' % (avg(ok), avg(mis)))
lines.append('')
for key in sorted(groups, key=lambda k: -len(groups[k])):
    items = groups[key]
    lines.append(f'### 真实={key[0]} 预测={key[1]}  （{len(items)} 条，平均回答长度 {avg(items):.1f} 词）')
    lines.append('')

# 每个类型输出全部样本的摘要（含概率）
order = ['prob_Ambivalent', 'prob_Clear_Non-Reply', 'prob_Clear_Reply']
for key in sorted(groups, key=lambda k: -len(groups[k])):
    items = groups[key]
    lines.append(f'{key[0]} -> {key[1]}  ({len(items)} 条)')
    for i, r in enumerate(items):
        q = r['question'].strip().replace('\n', ' ')
        a = r['interview_answer'].strip().replace('\n', ' ')
        w = wc(a)
        prob = '/'.join('%.3f' % float(r[k]) for k in order)
        lines.append(f'[{i}] {w}词  prob(A/CNR/CR)={prob}')
        lines.append(f'    Q: {q[:180]}')
        lines.append(f'    A: {a[:220]}')
    lines.append('')

with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines))
print('written', OUT, 'mis', len(mis), 'groups', len(groups))
