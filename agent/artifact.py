from __future__ import annotations

import json
import pathlib


def inspect_artifacts(output_dir: pathlib.Path) -> str:
    """Generate a compact post-execution artifact report."""
    reports = []
    for p in sorted(output_dir.iterdir()):
        if not p.is_file():
            continue
        item = [f"FILE: {p.name}", f"SIZE: {p.stat().st_size} bytes"]
        suffix = p.suffix.lower()
        try:
            if suffix in {".csv", ".parquet"}:
                import pandas as pd
                df = pd.read_csv(p) if suffix == ".csv" else pd.read_parquet(p)
                item.append(f"SHAPE: {df.shape}")
                item.append(f"COLUMNS: {list(df.columns)}")
                item.append(f"NULLS: {df.isna().sum().sum()}")
                numeric = df.select_dtypes(include="number")
                if not numeric.empty:
                    item.append(f"NUMERIC_RANGE: {numeric.agg(['min','max']).to_dict()}")
            elif suffix == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    item.append(f"JSON_KEYS: {list(obj.keys())[:30]}")
                elif isinstance(obj, list):
                    item.append(f"JSON_LENGTH: {len(obj)}")
            
        except Exception as e:
            item.append(f"INSPECTION_ERROR: {type(e).__name__}: {e}")
        reports.append("\n".join(item))
    return "\n\n".join(reports) or "NO_ARTIFACTS_FOUND"
