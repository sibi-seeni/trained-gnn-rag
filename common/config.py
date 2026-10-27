"""Config loading with single inheritance.

A config can say `base: configs/baseline_plain.yaml` and only list what it changes.
Nested dicts are merged key by key; anything else is replaced.
"""
import copy
import yaml


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_cfg(path, overrides=None):
    cfg = yaml.safe_load(open(path))
    if "base" in cfg:
        cfg = _merge(load_cfg(cfg.pop("base")), cfg)
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg
