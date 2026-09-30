# ReelRank — 基于 TMDB 的搜索 / 推荐 / 广告四层漏斗排序系统

以 **TMDB** 为数据源，经 **DuckDB + Pandas** 离线数仓与特征工程训练真实排序模型，在线请求依次经过
**召回 → 粗排 → 精排 → 重排** 四层漏斗，并在重排阶段完成 **广告竞价混排**（eCPM = pCTR × bid × 质量因子，GSP 二价扣费）。
提供 FastAPI 服务与 React 前端，含可逐层下钻的**漏斗调试面板**。

## 快速开始

```bash
# 1. 安装依赖（Python 3.10+，本项目在 3.14 上验证）
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt && ./.venv/bin/pip install -e .

# 2. 配置 TMDB 凭证（可选，缺失时使用 data/seed 内置样例快照）
cp .env.example .env        # 填入 TMDB_API_KEY（https://www.themoviedb.org/settings/api）

# 3. 一键跑通离线链路：采集 → 数仓 ETL → 特征 → 训练 → 产物落盘
bash scripts/run_pipeline.sh

# 4. 启动在线服务 + 前端
./.venv/bin/python -m reelrank.serving.api      # http://localhost:8000/docs
cd web && npm install && npm run dev            # http://localhost:5173
```

## 数据来源说明

| 数据 | 来源 | 用途 |
| --- | --- | --- |
| 影片元数据/演职员/关键词 | TMDB API v3（`/movie/popular`、`/discover/movie`、`/search/movie` + details/credits/keywords） | 内容理解、倒排索引、筛选 |
| Item-Item 相似图 | TMDB `/movie/{id}/similar`、`/movie/{id}/recommendations` + 主演/导演/关键词共现 | i2i 召回（真实关系，非随机） |
| 用户行为 | MovieLens `ml-latest-small`（`links.csv` 含 `tmdbId`，与 TMDB 精确对齐） | ItemCF 共现、CTR 训练标签、用户画像 |
| 用户行为（回退） | 固定种子合成日志（`behavior.source: synthetic`） | MovieLens 下载失败时保证链路可跑通 |
| 样例快照 | `data/seed/*.json` | 无 TMDB Key 时端到端 Demo |

> TMDB 无公开用户评分接口，因此行为标签来自 MovieLens（真实）或合成日志（标注来源，见 `data/warehouse/meta.json`）。

## 四层漏斗

| 阶段 | 做法 | 候选量 |
| --- | --- | --- |
| 召回 | 多路并行：BM25 倒排、SVD 向量 ANN、ItemCF（行为共现 + TMDB 相似图）、热门/规则 | ~6000 → 1000 |
| 粗排 | LogisticRegression 轻量预估 + 双塔内积 + 贝叶斯质量分融合 | 1000 → 300 |
| 精排 | `HistGradientBoostingClassifier`(GBDT) 预估 pCTR，融合质量/新鲜度多目标 | 300 → 50 |
| 重排 | MMR 多样性打散、同导演/同类型窗口去重、探索流量、质量门槛 | 50 → 20 |
| 广告 | eCPM 排序 → 广告位插入 → 频次/预算控制 → GSP 二价扣费 | 混入最终流 |

每次请求返回 `FunnelTrace`，记录各阶段候选 id、数量、耗时、分数分量与截断原因，前端调试面板直接消费。

## 目录结构

```
config/settings.yaml      # 采集/漏斗/广告/服务全部参数
src/reelrank/
  config.py               # 配置加载（settings.yaml + .env）
  data/                   # TMDB 采集客户端、行为数据源
  warehouse/              # DuckDB schema.sql 与 ETL
  features/               # 影片/用户特征工程
  models/                 # BM25 / SVD向量 / ItemCF / 粗排LR / 精排GBDT
  recall/ rank/ ads/      # 多路召回、四层排序、广告竞价混排
  serving/                # FastAPI 服务、FunnelTrace 编排
  eval/                   # Recall@K / NDCG@K / AUC / eCPM 评估
web/                      # React 前端（搜索 / 推荐 / 漏斗面板 / 广告看板）
scripts/                  # run_pipeline.sh、run_eval.py
```

## 评估

```bash
./.venv/bin/python scripts/run_eval.py     # 输出 Recall@K、NDCG@K、CTR AUC 与漏斗耗时报告
```

## 兼容性说明

本机 Python 3.14，刻意规避 `lightgbm`/`xgboost`/`faiss`/`implicit` 等无稳定 wheel 的库：
GBDT 用 sklearn 内建 `HistGradientBoostingClassifier`，向量检索用 Numpy 归一化暴力内积（万级规模毫秒级）。
# tmdb_lake_house_demo01
