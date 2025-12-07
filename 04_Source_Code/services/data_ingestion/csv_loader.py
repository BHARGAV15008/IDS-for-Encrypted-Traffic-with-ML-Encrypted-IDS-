from typing import Optional
import pandas as pd


def load_csv_dataset(path: str, label_column: Optional[str] = None) -> pd.DataFrame:

    data = pd.read_csv(path)
    if label_column is not None and label_column not in data.columns:
        raise ValueError(f"label_column '{label_column}' not in dataset")
    return data


