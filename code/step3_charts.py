# -*- coding: utf-8 -*-
"""第 3 步：画实验报告要的两张图。
图1：清晰度标签分布（柱状图）
图2：文本长度分布（回答长度直方图 + 按标签的箱线图）
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
os.makedirs(BASE + 'results', exist_ok=True)   # 确保输出目录存在（全新环境首次运行）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = BASE + 'data/hf'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 调色板（可视化规范里验证过的颜色）
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SURFACE, INK, MUTED, GRID = '#ffffff', '#0b0b0b', '#898781', '#e1e0d9'

ds = load_dataset('ailsntua/QEvasion')
train = pd.DataFrame(ds['train'])
train['a_len'] = train['interview_answer'].str.split().str.len()
train['q_len'] = train['question'].str.split().str.len()

order = ['Clear Reply', 'Ambivalent', 'Clear Non-Reply']
colors = {'Clear Reply': BLUE, 'Ambivalent': ORANGE, 'Clear Non-Reply': AQUA}

# 图1：标签分布
vc = train['clarity_label'].value_counts().reindex(order)
fig, ax = plt.subplots(figsize=(6, 4.2))
bars = ax.bar(order, vc.values, color=[colors[c] for c in order], width=0.55)
for b, n in zip(bars, vc.values):
    ax.text(b.get_x() + b.get_width()/2, n, f'{n} 条\n({n/len(train)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, color=INK)
ax.set_title('QEvasion 训练集清晰度标签分布', fontsize=13, color=INK, pad=12)
ax.set_ylabel('样本数', fontsize=11, color=MUTED)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED)
ax.set_ylim(0, vc.max() * 1.25)
fig.tight_layout()
fig.savefig(BASE + 'results/fig1_label_dist.png', dpi=150, facecolor=SURFACE)
plt.close(fig)

# 图2：长度分布（两栏）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# 左：回答长度直方图（截断到 99 分位，避免极长样本把图拉扁）
cap = np.percentile(train['a_len'], 99)
data = train['a_len'].clip(upper=cap)
ax1.hist(data, bins=40, color=BLUE, alpha=0.85)
ax1.axvline(train['a_len'].mean(), color=ORANGE, lw=2, ls='--', label='均值')
ax1.set_title('回答长度分布（interview_answer）', fontsize=12, color=INK)
ax1.set_xlabel('单词数', fontsize=10, color=MUTED)
ax1.set_ylabel('样本数', fontsize=10, color=MUTED)
ax1.legend(frameon=False)
ax1.spines[['top', 'right']].set_visible(False)
ax1.yaxis.grid(True, color=GRID)
ax1.set_axisbelow(True)
ax1.tick_params(colors=MUTED)

# 右：按标签的回答长度箱线图
boxdata = [train.loc[train['clarity_label'] == c, 'a_len'].values for c in order]
bp = ax2.boxplot(boxdata, tick_labels=order, patch_artist=True, showfliers=False)
for patch, c in zip(bp['boxes'], order):
    patch.set_facecolor(colors[c])
    patch.set_alpha(0.7)
ax2.set_title('回答长度按标签比较', fontsize=12, color=INK)
ax2.set_ylabel('单词数', fontsize=10, color=MUTED)
ax2.spines[['top', 'right']].set_visible(False)
ax2.yaxis.grid(True, color=GRID)
ax2.set_axisbelow(True)
ax2.tick_params(colors=MUTED)

fig.tight_layout()
fig.savefig(BASE + 'results/fig2_length_dist.png', dpi=150, facecolor=SURFACE)
plt.close(fig)

print('图表已保存：results/fig1_label_dist.png, results/fig2_length_dist.png')
