# -*- coding: utf-8 -*-
"""第 14 步：从 baselineB_predictions.csv 里挑典型误判样本（用于表5）。

按混淆类型分组，每组打印几条候选（问题/回答截断、gold、pred、三类概率），
方便人工挑出最能说明问题的 5 条。
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
import pandas as pd

df = pd.read_csv(BASE + 'results/baselineB_predictions.csv', encoding='utf-8-sig')
df['a_len'] = df['interview_answer'].astype(str).str.split().str.len()
mis = df[df['gold'] != df['prediction']].copy()

PROB = {'prob_Clear_Reply': 'ClearReply', 'prob_Ambivalent': 'Ambivalent', 'prob_Clear_Non-Reply': 'NonReply'}

def show(row, label):
    q = row['question'][:60]
    a = row['interview_answer'][:120].replace('\n', ' ')
    probs = ' | '.join(f"{short}:{row[p]:.3f}" for p, short in PROB.items())
    print(f"\n[{label}] gold={row['gold']} -> pred={row['prediction']}  (回答{row['a_len']}词)")
    print(f"  Q: {q}...")
    print(f"  A: {a}...")
    print(f"  prob: {probs}")

# 1) 明确拒答：gold=Clear Non-Reply, pred=Clear Reply
g = mis[(mis['gold'] == 'Clear Non-Reply') & (mis['prediction'] == 'Clear Reply')]
print("=" * 70)
print(f"类型1 明确拒答被当回答  ({len(g)}条)")
for _, r in g.head(4).iterrows():
    show(r, '明确拒答')

# 2) 声称无知/隐式：gold=Clear Non-Reply, pred=Ambivalent
g = mis[(mis['gold'] == 'Clear Non-Reply') & (mis['prediction'] == 'Ambivalent')]
print("=" * 70)
print(f"类型2 声称无知被当含糊  ({len(g)}条)")
for _, r in g.head(4).iterrows():
    show(r, '声称无知')

# 3) 长而空泛：gold=Clear Non-Reply, pred=Ambivalent，取回答最长的
g = mis[(mis['gold'] == 'Clear Non-Reply') & (mis['prediction'] == 'Ambivalent')].sort_values('a_len', ascending=False)
print("=" * 70)
print(f"类型3 长而空泛（回答最长）")
for _, r in g.head(3).iterrows():
    show(r, '长而空泛')

# 4) 主题转移：gold=Ambivalent 或 Clear Non-Reply 的边界，找回答里含转移词
g = mis[mis['interview_answer'].astype(str).str.contains('pig|joke|anyway|speaking of', case=False, na=False)]
print("=" * 70)
print(f"类型4 主题转移候选（含 pig/joke/anyway）")
for _, r in g.head(4).iterrows():
    show(r, '主题转移')

# 5) 数据噪声：回答最短
g = mis.sort_values('a_len').head(5)
print("=" * 70)
print(f"类型5 数据噪声（回答最短）")
for _, r in g.iterrows():
    show(r, '数据噪声')

print("\n\n== 全部误判按混淆类型的数量 ==")
print(mis.groupby(['gold', 'prediction']).size())
