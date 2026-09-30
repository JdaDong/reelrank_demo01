"""统一日志出口：一行/请求记录漏斗各层候选量与耗时。"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def configure(level: str = "INFO") -> None:
    logging.getLogger().setLevel(level.upper())
    for name in ("reelrank", "uvicorn", "fastapi"):
        get_logger(name).setLevel(level.upper())
