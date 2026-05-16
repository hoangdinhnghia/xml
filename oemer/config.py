"""Simple config loader for Oemer post-processing and inference.
Reads config.json at repository root if present, otherwise provides defaults.
"""
import json
import os

DEFAULTS = {
    "confidence_threshold": 0.6,
    "low_conf_min_area": 64,
    "merge_method": "gaussian",
    "cache_dir": "output/cache",
}

def load_config():
    root = os.getcwd()
    cfg_path = os.path.join(root, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            cfg = dict(DEFAULTS)
            cfg.update(data)
            return cfg
        except Exception:
            return DEFAULTS
    return DEFAULTS

CONFIG = load_config()
