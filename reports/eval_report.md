# ReelRank 离线评估报告

- 生成时间：2026-09-30T04:22:05.049584+00:00
- 数据来源：seed_snapshot / 行为来源：synthetic

## 推荐场景

### 各路召回 Recall@K（训练段历史触发，测试段正例为标签）

| 指标 | 值 |
| --- | --- |
| itemcf | 0.59 |
| vector | 0.5527 |
| hot | 0.2658 |
| merged | 0.59 |

### 最终排序结果（对照随机基线）

| 指标 | 值 |
| --- | --- |
| ndcg@10 | 0.1351 |
| ndcg@20 | 0.1568 |
| ndcg@50 | 0.1568 |
| recall@10 | 0.1339 |
| recall@20 | 0.1871 |
| recall@50 | 0.1871 |
| precision@5 | 0.0833 |
| baseline_recall@10 | 0.0167 |
| baseline_recall@20 | 0.0333 |
| baseline_recall@50 | 0.0833 |

## 搜索场景

### 各路召回 Recall@K

| 指标 | 值 |
| --- | --- |
| bm25 | 0.3875 |
| vector | 0.5208 |
| hot | 0.1528 |
| merged | 0.3875 |

### 最终排序结果

| 指标 | 值 |
| --- | --- |
| ndcg@10 | 0.1638 |
| ndcg@20 | 0.1775 |
| ndcg@50 | 0.1775 |
| precision@3 | 0.1722 |

## 排序模型

### 粗排 / 精排

| 指标 | 值 |
| --- | --- |
| coarse_auc | 0.7569 |
| coarse_auc_by_scene | {'recommend': 0.7426, 'search': 0.851} |
| fine_auc | 0.9194 |
| fine_gauc | 0.9081 |
| samples | 800 |

## 广告

### 竞价与收入

| 指标 | 值 |
| --- | --- |
| requests | 20 |
| impressions | 60 |
| avg_ecpm | 116.2 |
| avg_price | 4.0329 |
| revenue | 5.5979 |
| rpm | 93.3 |

## 漏斗耗时（ms）

### 各层耗时

| 指标 | 值 |
| --- | --- |
| recall | avg 1.95 / p95 4.0 |
| coarse | avg 22.97 / p95 31.99 |
| fine | avg 22.9 / p95 31.63 |
| rerank | avg 7.29 / p95 9.07 |
| ads | avg 11.59 / p95 14.94 |
| total | avg 68.33 / p95 83.63 |
