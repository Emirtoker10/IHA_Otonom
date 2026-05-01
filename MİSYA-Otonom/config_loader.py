"""
config_loader.py
Merkezi YAML yapılandırma yükleyici.
Tüm modüller bu modül üzerinden config okur.
"""

import yaml
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"
_config_cache: dict = {}


def load_config(path: str | Path = _CONFIG_PATH) -> dict:
    """YAML config dosyasını yükler ve döndürür (önbellekli)."""
    global _config_cache
    if _config_cache:
        return _config_cache

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError("Config dosyası boş veya geçersiz YAML.")

    _config_cache = cfg
    logger.info(f"Config yüklendi: {path}")
    return cfg


def get(section: str, key: str = None, default=None):
    """
    Belirli bir config anahtarına kısayol erişimi.

    Örnek:
        get("yolo", "confidence_threshold")  → 0.55
        get("mavlink")                        → {'connection_string': ..., ...}
    """
    cfg = load_config()
    section_data = cfg.get(section, {})
    if key is None:
        return section_data
    return section_data.get(key, default)


def reload():
    """Config önbelleğini temizleyip yeniden yükler."""
    global _config_cache
    _config_cache = {}
    return load_config()
