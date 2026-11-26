# -*- coding: utf-8 -*-
"""第 2 步：探索性数据分析（EDA）。
统计：类别分布、长度分布、president 分布、缺失值。结果写入 results/eda_summary.txt（UTF-8）。
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
os.makedirs(BASE + 'results', exist_ok=True)   # 确保输出目录存在（全新环境首次运行）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = BASE + 'data/hf'

import pandas as pd
from datasets import load_dataset

ds = load_dataset("ailsntua/QEvasion")
train = pd.DataFrame(ds['train'])
test = pd.DataFrame(ds['test'])

lines = []
def p(s=""):
    lines.append(s)

p("=" * 60)
p("QEvasion 数据集 EDA 摘要")
p("=" * 60)

# 1) 样本数
p("\n【1】样本数")
p(f"  train = {len(train)}   test = {len(test)}   合计 = {len(train)+len(test)}")

# 2) 缺失值
p("\n【2】关键字段缺失值（训练集）")
for col in ['question', 'interview_answer', 'clarity_label', 'evasion_label', 'president']:
    p(f"  {col}: 缺失 {train[col].isna().sum()} 条")

# 3) clarity_label 分布
p("\n【3】清晰度标签分布（clarity_label，3类）—— 这是你要预测的目标")
for split_name, df in [('train', train), ('test', test)]:
    vc = df['clarity_label'].value_counts()
    p(f"  [{split_name}]")
    for label, n in vc.items():
        p(f"     {label}: {n} 条  ({n/len(df)*100:.1f}%)")

# 4) evasion_label 分布（细粒度，9类）
p("\n【4】细粒度标签分布（evasion_label）")
vc = train['evasion_label'].value_counts()
for label, n in vc.items():
    p(f"  {label}: {n} 条  ({n/len(train)*100:.1f}%)")

# 5) 长度分布
p("\n【5】文本长度（按空格分词数 = 大致单词数）")
train['q_len'] = train['question'].str.split().str.len()
train['a_len'] = train['interview_answer'].str.split().str.len()
p(f"  问题 question 长度：平均 {train['q_len'].mean():.1f}  中位数 {train['q_len'].median():.0f}  最长 {train['q_len'].max()}")
p(f"  回答 answer 长度：平均 {train['a_len'].mean():.1f}  中位数 {train['a_len'].median():.0f}  最长 {train['a_len'].max()}")
p("  按 clarity_label 看回答平均长度：")
for label, g in train.groupby('clarity_label')['a_len']:
    p(f"     {label}: 平均 {g.mean():.1f}")

# 6) president 分布（查数据泄漏）
p("\n【6】president 分布（查同源采访泄漏）")
p(f"  训练集共有 {train['president'].nunique()} 位不同的总统")
for name, n in train['president'].value_counts().head(6).items():
    p(f"     {name}: {n} 条")
# 测试集总统是否也出现在训练集
p(f"  测试集中有 {test['president'].nunique()} 位总统，其中 {test['president'].isin(train['president']).sum()}/{len(test)} 条测试样本的总统也出现在训练集")

with open(BASE + 'results/eda_summary.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))

print("EDA 完成，结果已保存到 results/eda_summary.txt")
print(f"train={len(train)} test={len(test)}")
print(train['clarity_label'].value_counts().to_dict())
