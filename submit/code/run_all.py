# -*- coding: utf-8 -*-
"""一键复现入口。

用法（在 code/ 目录下）：
    python run_all.py           # 跑核心必做链路（约几分钟，不含微调/迁移）
    python run_all.py --full    # 额外跑迁移、微调、消融、数据效率（较慢）

脚本按顺序执行；核心步骤失败会停止，可选步骤失败仅警告。
"""
import os
import subprocess
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
for _d in ('data/hf', 'results', 'predictions', 'models'):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)   # 预创建输出目录，全新环境可直接运行


CORE = [
    'step1_download.py',       # 下载数据
    'step2_eda.py',            # EDA
    'step3_charts.py',         # 图1/图2
    'step4_baselineA.py',      # TF-IDF + LR
    'step5_baselineB.py',      # 冻结 E5-small + LR
    'step10_mlp.py',           # 生成 e5small_emb_*.npy（step12 依赖）
    'step12_proba.py',         # 补三类概率列
    'step14_find_misclass.py', # 挑典型误判样本
]

OPTIONAL = [
    'step16_ablation.py',          # 指令前缀 [Q]/[A] 消融
    'step13_data_efficiency.py',   # 数据效率 10%/30%/100%
    'step11_qnli.py',              # QNLI 跨场景迁移
    'step9_finetune.py',           # 对比微调 E5-small
]


def run_one(name, required):
    path = os.path.join(CODE, name)
    if not os.path.exists(path):
        print(f'[跳过] 找不到 {name}')
        return required is False
    print(f'\n{"=" * 64}\n>>> 运行 {name}\n{"=" * 64}')
    r = subprocess.run([sys.executable, path])
    ok = r.returncode == 0
    if not ok:
        tag = '失败' if required else '警告'
        print(f'[{tag}] {name} 退出码 {r.returncode}')
    return ok


def main():
    full = '--full' in sys.argv
    scripts = CORE + (OPTIONAL if full else [])
    for name in scripts:
        required = name in CORE
        ok = run_one(name, required)
        if required and not ok:
            print('\n核心步骤失败，已停止。')
            sys.exit(1)
    print('\n全部完成。')
    if not full:
        print('提示：迁移/微调/消融/数据效率等加分实验请用 `python run_all.py --full` 运行。')


if __name__ == '__main__':
    main()
