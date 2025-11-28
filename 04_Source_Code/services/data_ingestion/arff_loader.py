from pathlib import Path
from typing import List, Tuple
import pandas as pd
from scipy.io import arff
import numpy as np


def _decode_and_clean(df: pd.DataFrame) -> pd.DataFrame:

    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].apply(lambda v: v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else v)
            except Exception:
                pass
        # Normalize empty strings and '?' to NaN
        if df[col].dtype == object:
            df[col] = df[col].replace({"": np.nan, "?": np.nan})
    return df


def load_arff_file(path: str) -> pd.DataFrame:

    try:
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        return _decode_and_clean(df)
    except Exception:
        # Fallback to liac-arff for more tolerant parsing
        import arff as liac_arff  # type: ignore
        try:
            with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                arff_obj = liac_arff.load(f)
            df = pd.DataFrame(arff_obj['data'], columns=[attr[0] for attr in arff_obj['attributes']])
            return _decode_and_clean(df)
        except Exception:
            # Ultimate fallback: tolerant manual parser
            return _parse_arff_tolerant(path)


def _parse_arff_tolerant(path: str) -> pd.DataFrame:

    relation_seen = False
    attributes: List[Tuple[str, str]] = []
    data_rows: List[List[str]] = []
    in_data = False

    with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('%'):
                continue
            low = line.lower()
            if low.startswith('@relation'):
                relation_seen = True
                continue
            if low.startswith('@attribute'):
                # Format: @attribute name type
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[1].strip().strip("'\"")
                    attr_type = ' '.join(parts[2:])
                    attributes.append((name, attr_type))
                continue
            if low.startswith('@data'):
                in_data = True
                continue
            if in_data:
                # Split CSV-like, tolerant
                # Handle simple cases without embedded commas inside quotes
                row = [c.strip().strip('"\'') for c in line.split(',')]
                data_rows.append(row)

    # Build DataFrame with aligned columns
    col_names = [a[0] for a in attributes] if attributes else []
    df = pd.DataFrame(data_rows, columns=col_names if col_names else None)
    df = _decode_and_clean(df)
    return df



def load_multiple_arff(paths: List[str]) -> pd.DataFrame:

    frames: List[pd.DataFrame] = []
    for p in paths:
        df = load_arff_file(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def find_arff_files_in_dirs(directories: List[str]) -> List[str]:

    files: List[str] = []
    for d in directories:
        for f in Path(d).glob('*.arff'):
            files.append(str(f))
    return files


