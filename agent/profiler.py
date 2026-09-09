"""Lightweight deterministic input profiler for Track-1 agents.

This module intentionally exposes only data/schema observations and never task
solutions or grader information.
"""
from __future__ import annotations

import json
import pathlib


def _safe(v):
    try:
        return v.item()
    except Exception:
        return str(v)


def profile_file(path: pathlib.Path, max_rows: int = 3) -> str:
    suffix = path.suffix.lower()
    lines = [f"FILE: {path.name}", f"SIZE: {path.stat().st_size} bytes"]
    try:
        if suffix in {".csv", ".parquet"}:
            import pandas as pd
            df = pd.read_csv(path) if suffix == ".csv" else pd.read_parquet(path)
            lines += [f"SHAPE: {df.shape}", "COLUMNS:"]
            for c, t in df.dtypes.items():
                lines.append(f"- {c}: {t}")
            lines.append("SAMPLE:")
            lines.append(df.head(max_rows).to_string())
            lines.append("NULLS:")
            lines.append(df.isna().sum().to_string())
        elif suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                lines.append("KEYS: " + ", ".join(map(str, obj.keys())))
            lines.append("SAMPLE: " + json.dumps(obj if isinstance(obj, (dict, list)) else str(obj) , default=_safe)[:1000])
        elif suffix in {".npy", ".npz"}:
            import numpy as np
            data = np.load(path)
            if suffix == ".npy":
                lines.append(f"ARRAY: shape={data.shape}, dtype={data.dtype}")
            else:
                lines.append("KEYS: " + ", ".join(data.files))
        else:
            lines.append("TYPE: unsupported for deep profiling")
    except Exception as e:
        lines.append(f"PROFILE_ERROR: {type(e).__name__}: {e}")
    return "\n".join(lines)


def profile_inputs(task_dir: pathlib.Path) -> str:
    outputs = []
    for p in sorted(task_dir.rglob("*")):
        if p.is_file() and ".git" not in p.parts and "checks" not in p.parts and p.name not in {"card.toml", "manifest.json"}:
            outputs.append(profile_file(p))
    return "\n\n".join(outputs) or "(no input files found)"
