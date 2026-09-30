#!/usr/bin/env bash
# 一键跑通：采集/建仓 → 行为对齐与特征工程 → 索引与排序模型训练（可选评估）
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

STEP=0
run() {
  STEP=$((STEP + 1))
  echo ""
  echo "==> [$STEP/3] $1"
  shift
  "$PY" "$@"
}

echo "ReelRank 一键 Pipeline - Python ${PY}"
run "数据采集与 DuckDB 建仓（无 TMDB Key 时自动使用样例快照）" -m reelrank.warehouse.etl
run "行为对齐与特征工程" -m reelrank.features.pipeline
run "召回索引与排序模型训练" -m reelrank.train.train_offline

if [[ "${1:-}" == "--eval" ]]; then
  echo ""
  echo "==> [附加] 离线评估"
  "$PY" scripts/run_eval.py
fi

echo ""
echo "完成。启动服务： $PY -m reelrank.serving.api"
