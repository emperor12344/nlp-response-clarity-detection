# -*- coding: utf-8 -*-
"""第 1 步：下载 QEvasion 数据集，并打印它的结构。
国内 HuggingFace 官网连不上，所以走镜像 hf-mirror.com。
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = BASE + 'data/hf'

from datasets import load_dataset

print("正在下载 QEvasion 数据集（首次会慢一点，之后有缓存）...\n")
ds = load_dataset("ailsntua/QEvasion")
print("下载完成！\n")

print("=" * 50)
print("数据集里有哪些划分（split）：")
for name in ds.keys():
    print(f"  · {name}: {len(ds[name])} 条")

print("\n" + "=" * 50)
print("每条数据有哪些字段（列）：")
print(ds['train'].features)

print("\n" + "=" * 50)
print("第一条数据长什么样（看看真实内容）：")
first = ds['train'][0]
for k, v in first.items():
    s = str(v)
    if len(s) > 100:
        s = s[:100] + "……"
    print(f"  {k} = {s}")
