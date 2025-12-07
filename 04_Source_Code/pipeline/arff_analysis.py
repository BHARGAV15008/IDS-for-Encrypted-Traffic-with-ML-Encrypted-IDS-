from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from services.data_ingestion.arff_loader import find_arff_files_in_dirs, load_multiple_arff


def analyze_single_dir(dir_path: str) -> Dict:

    files = find_arff_files_in_dirs([dir_path])
    if not files:
        return {"directory": dir_path, "error": "No ARFF files"}
    df = load_multiple_arff(files)

    # Basic schema
    dtypes = df.dtypes.astype(str).to_dict()
    n_rows, n_cols = df.shape
    null_counts = df.isna().sum().to_dict()

    # Label detection
    label_candidates = [c for c in df.columns if str(c).lower() in ['class', 'label', 'target', 'attack_cat', 'attack', 'category']]
    label_col = label_candidates[0] if label_candidates else df.columns[-1]
    labels = df[label_col].astype(str)
    label_counts = labels.value_counts().to_dict()

    # Numeric coverage
    X = df.drop(columns=[label_col])
    X_num = X.apply(pd.to_numeric, errors='coerce')
    non_numeric_cols = [c for c in X.columns if not np.issubdtype(X_num[c].dtype, np.number)]

    return {
        "directory": dir_path,
        "files": files,
        "rows": n_rows,
        "cols": n_cols,
        "label_col": label_col,
        "label_counts": label_counts,
        "dtypes": dtypes,
        "null_counts": null_counts,
        "non_numeric_cols": non_numeric_cols,
        "feature_cols": [c for c in X.columns]
    }


def analyze_arff_dirs(dirs: List[str]) -> Dict:

    per_dir = [analyze_single_dir(d) for d in dirs]

    # Feature set comparisons
    feature_sets = {r["directory"]: set(r.get("feature_cols", [])) for r in per_dir if "feature_cols" in r}
    common_features = set.intersection(*(s for s in feature_sets.values())) if feature_sets else set()
    union_features = set.union(*(s for s in feature_sets.values())) if feature_sets else set()

    # Label harmonization suggestion
    labels_map = {r["directory"]: set(map(str, r.get("label_counts", {}).keys())) for r in per_dir if "label_counts" in r}
    common_labels = set.intersection(*(s for s in labels_map.values())) if labels_map else set()
    union_labels = set.union(*(s for s in labels_map.values())) if labels_map else set()

    return {
        "summary": {
            "common_features_count": len(common_features),
            "union_features_count": len(union_features),
            "common_labels_count": len(common_labels),
            "union_labels_count": len(union_labels)
        },
        "common_features": sorted(common_features),
        "union_features": sorted(union_features),
        "common_labels": sorted(common_labels),
        "union_labels": sorted(union_labels),
        "per_dir": per_dir
    }


def write_report(report: Dict, out_path: str) -> None:

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        f.write("=== ARFF Dataset Analysis Report ===\n\n")
        f.write(f"Common features: {report['summary']['common_features_count']}\n")
        f.write(f"Union features: {report['summary']['union_features_count']}\n")
        f.write(f"Common labels: {report['summary']['common_labels_count']}\n")
        f.write(f"Union labels: {report['summary']['union_labels_count']}\n\n")

        f.write("-- Common Labels --\n")
        f.write(", ".join(report.get('common_labels', [])) + "\n\n")

        f.write("-- Per Directory --\n")
        for r in report.get('per_dir', []):
            f.write(f"Directory: {r.get('directory')}\n")
            if 'error' in r:
                f.write(f"  Error: {r['error']}\n\n")
                continue
            f.write(f"  Rows: {r.get('rows')}  Cols: {r.get('cols')}\n")
            f.write(f"  Label Column: {r.get('label_col')}\n")
            f.write("  Label Counts:\n")
            for k, v in r.get('label_counts', {}).items():
                f.write(f"    {k}: {v}\n")
            f.write(f"  Non-numeric Features: {len(r.get('non_numeric_cols', []))}\n")
            if r.get('non_numeric_cols'):
                f.write("    " + ", ".join(r['non_numeric_cols']) + "\n")
            f.write("\n")


