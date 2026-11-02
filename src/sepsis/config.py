"""Tek yapılandırma kaynağı: configs/default.yaml (SEPSIS_CONFIG ortam değişkeniyle değiştirilebilir)."""
import os
from pathlib import Path
import yaml

def load_config(path: str | None = None) -> dict:
    path = path or os.environ.get("SEPSIS_CONFIG", "configs/default.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: str(Path(v).expanduser()) for k, v in cfg["paths"].items()}
    return cfg

def direction_slug(direction: str) -> str:
    """'A->B' -> 'A_B' (dosya adları için)."""
    return direction.replace("->", "_")
