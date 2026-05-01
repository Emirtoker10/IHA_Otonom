"""
logger_setup.py
Merkezi loglama yapılandırması.
main.py başında bir kez çağrılır; diğer modüller logging.getLogger kullanır.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from config_loader import get


def setup_logging() -> logging.Logger:
    """
    Rotating file + console loglama ayarlar.
    Returns root logger.
    """
    cfg = get("logging")
    level_str: str = cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    log_dir = Path(cfg.get("log_dir", "logs/"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "vtol_sar.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)-25s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=cfg.get("max_bytes", 10_485_760),
        backupCount=cfg.get("backup_count", 5),
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    if not root.handlers:
        root.addHandler(console)
        root.addHandler(file_handler)

    return root
