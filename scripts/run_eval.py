"""离线评估 CLI：python scripts/run_eval.py [--users 120] [--queries 120] [--requests 20]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reelrank.eval.offline_eval import run_evaluation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="ReelRank 离线评估")
    parser.add_argument("--users", type=int, default=120, help="推荐场景采样用户数")
    parser.add_argument("--queries", type=int, default=120, help="搜索场景采样 query 数")
    parser.add_argument("--requests", type=int, default=20, help="在线请求次数（广告与耗时）")
    parser.add_argument("--topk-route", type=int, default=100, help="单路召回评估的截断 K")
    args = parser.parse_args()

    report = run_evaluation(
        sample_users=args.users,
        sample_queries=args.queries,
        requests=args.requests,
        topk_route=args.topk_route,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
