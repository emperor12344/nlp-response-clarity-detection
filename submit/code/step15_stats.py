# -*- coding: utf-8 -*-
"""第 15 步：补表1 需要的数据统计（原任务 + 迁移任务的样本数/缺失/平均长度）。"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = BASE + 'data/hf'

import pandas as pd
from datasets import load_dataset

def stats(df, q_col, a_col, name):
    n = len(df)
    miss = int(df[[q_col, a_col]].isna().any(axis=1).sum())
    q_len = df[q_col].astype(str).str.split().str.len()
    a_len = df[a_col].astype(str).str.split().str.len()
    print(f"{name}: 样本数={n}, 缺失样本={miss}, 平均问题长度={q_len.mean():.1f}, 平均回答长度={a_len.mean():.1f}")

# 原任务 QEvasion
qe = load_dataset('ailsntua/QEvasion')
stats(pd.DataFrame(qe['train']), 'question', 'interview_answer', 'QEvasion-train')
stats(pd.DataFrame(qe['test']), 'question', 'interview_answer', 'QEvasion-test')

# 迁移任务 QNLI（train 抽样 10000 = 与 step11/13 一致；test = 官方 validation）
qnli = load_dataset('nyu-mll/glue', 'qnli')
tr = qnli['train'].to_pandas()
from sklearn.model_selection import train_test_split
import numpy as np
idx = np.arange(len(tr))
tr_idx, _ = train_test_split(idx, train_size=10000, stratify=tr['label'].values, random_state=42)
sub = tr.iloc[tr_idx]
stats(sub.reset_index(drop=True), 'question', 'sentence', 'QNLI-train(抽样10000)')
stats(qnli['validation'].to_pandas(), 'question', 'sentence', 'QNLI-test(validation)')
