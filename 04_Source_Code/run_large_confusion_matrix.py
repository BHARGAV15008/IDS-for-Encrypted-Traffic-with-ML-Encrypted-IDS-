import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import sys
import importlib.util

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Dynamically load PerformanceVisualizer from its file path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISUALIZER_PATH = PROJECT_ROOT / "05_Evaluation" / "04_Visualization_Scripts" / "performanceVisualizer.py"

spec = importlib.util.spec_from_file_location("performanceVisualizer", VISUALIZER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load performanceVisualizer from {VISUALIZER_PATH}")
performanceVisualizer = importlib.util.module_from_spec(spec)
sys.modules["performanceVisualizer"] = performanceVisualizer
spec.loader.exec_module(performanceVisualizer)

PerformanceVisualizer = performanceVisualizer.PerformanceVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_large_dataset(csv_path: str, n_rows: int = 1_000_000) -> pd.DataFrame:
    """Load up to n_rows rows from a large CSV dataset."""
    logger.info(f"Loading up to {n_rows} rows from {csv_path} ...")
    df = pd.read_csv(csv_path, nrows=n_rows)
    logger.info(f"Loaded shape: {df.shape}")
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Basic preprocessing: drop non-useful columns, encode label, keep numeric features."""
    # Drop known non-feature columns if present
    drop_cols = [
        "Flow ID",
        " Source IP",
        " Destination IP",
        " Timestamp",
        "SimillarHTTP",
        " Inbound",
        " External IP",
    ]
    existing_drop = [c for c in drop_cols if c in df.columns]
    if existing_drop:
        logger.info(f"Dropping columns: {existing_drop}")
        df = df.drop(columns=existing_drop)

    # Label encoding: benign vs attack
    label_col_candidates = ["Label", " label", " LABEL"]
    label_col = None
    for c in label_col_candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise KeyError(f"Could not find label column in columns: {df.columns[:10]}")

    logger.info(f"Using label column: {label_col}")
    y_raw = df[label_col].astype(str)
    benign_values = {"0", "BENIGN", "benign", "Normal", "normal"}
    y = (~y_raw.isin(benign_values)).astype(int).to_numpy()

    # Drop label column from features
    df = df.drop(columns=[label_col])

    # Keep only numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    logger.info(f"Using {numeric_df.shape[1]} numeric features out of {df.shape[1]} total columns")

    # Fill NaNs and infs
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return numeric_df, y


def train_and_evaluate_streaming(
    csv_path: str,
    output_dir: Path,
    chunk_size: int = 200_000,
    max_test: int = 1_000_000,
) -> None:
    """Train and evaluate on a very large CSV using chunked streaming.

    This avoids loading all rows into memory at once by:
      - reading the CSV in chunks,
      - training an incremental (SGD) classifier with partial_fit,
      - building a bounded test set buffer (up to max_test samples).
    """
    logger.info(
        f"Streaming training from {csv_path} with chunk_size={chunk_size}, max_test={max_test}"
    )

    rng = np.random.default_rng(42)
    classes = np.array([0, 1], dtype=int)

    # Incremental classifier; will be created after first training chunk so we can set class-balanced weights
    clf = None

    feature_cols = None
    test_X_parts: list[pd.DataFrame] = []
    test_y_parts: list[np.ndarray] = []
    test_count = 0
    first_fit = True

    drop_cols = [
        "Flow ID",
        " Source IP",
        " Destination IP",
        " Timestamp",
        "SimillarHTTP",
        " Inbound",
        " External IP",
    ]
    label_col_candidates = ["Label", " label", " LABEL"]

    for chunk_idx, df in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        logger.info(f"Processing chunk {chunk_idx} with shape {df.shape}")

        # Drop non-feature columns if present
        existing_drop = [c for c in drop_cols if c in df.columns]
        if existing_drop:
            df = df.drop(columns=existing_drop)

        # Find label column
        label_col = None
        for c in label_col_candidates:
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            # Skip chunk if no label column (should not happen for processed datasets)
            logger.warning(
                f"Chunk {chunk_idx} has no label column among {label_col_candidates}; skipping."
            )
            continue

        # Binary label encoding: benign vs attack
        y_raw = df[label_col].astype(str)
        benign_values = {"0", "BENIGN", "benign", "Normal", "normal"}
        y_chunk = (~y_raw.isin(benign_values)).astype(int).to_numpy()

        # Drop label from features
        df = df.drop(columns=[label_col])

        # Keep only numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            logger.warning(f"Chunk {chunk_idx} has no numeric features; skipping.")
            continue

        # Establish and align feature columns across chunks
        if feature_cols is None:
            feature_cols = list(numeric_df.columns)
            logger.info(f"Using {len(feature_cols)} numeric features: {feature_cols[:5]}...")
        else:
            # Ensure all known feature columns exist; fill missing with 0
            for col in feature_cols:
                if col not in numeric_df.columns:
                    numeric_df[col] = 0.0
            numeric_df = numeric_df[feature_cols]

        # Replace NaNs / infs
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Random train/test split within the chunk (approx. 80/20)
        m = len(y_chunk)
        if m == 0:
            continue
        mask = rng.random(m) < 0.8
        X_train_chunk = numeric_df[mask]
        y_train_chunk = y_chunk[mask]
        X_test_chunk = numeric_df[~mask]
        y_test_chunk = y_chunk[~mask]

        if len(y_train_chunk) == 0:
            continue

        # Incremental training with class-balanced weights inferred from first training chunk
        if clf is None:
            # Estimate class weights similar to 'balanced': n_samples / (n_classes * n_samples_per_class)
            unique, counts = np.unique(y_train_chunk, return_counts=True)
            total = counts.sum()
            weight_dict = {int(cls): float(total / (len(unique) * cnt)) for cls, cnt in zip(unique, counts)}
            logger.info(f"Initial class weights from first chunk: {weight_dict}")

            clf = SGDClassifier(
                loss="log_loss",
                class_weight=weight_dict,
                random_state=42,
                n_jobs=-1,
            )
            clf.partial_fit(X_train_chunk, y_train_chunk, classes=classes)
            first_fit = False
        else:
            clf.partial_fit(X_train_chunk, y_train_chunk)

        # Build bounded test buffer
        if len(y_test_chunk) > 0 and test_count < max_test:
            remaining = max_test - test_count
            if len(y_test_chunk) > remaining:
                idx = rng.choice(len(y_test_chunk), size=remaining, replace=False)
                X_add = X_test_chunk.iloc[idx]
                y_add = y_test_chunk[idx]
            else:
                X_add = X_test_chunk
                y_add = y_test_chunk

            test_X_parts.append(X_add)
            test_y_parts.append(y_add)
            test_count += len(y_add)
            logger.info(f"Accumulated test samples: {test_count}")

    if first_fit:
        raise RuntimeError("No data was used for training; check input CSV and label column.")

    if not test_X_parts:
        raise RuntimeError("No test samples were collected; cannot compute confusion matrix.")

    X_test = pd.concat(test_X_parts, axis=0)
    y_test = np.concatenate(test_y_parts, axis=0)
    logger.info(f"Final test set shape: {X_test.shape}, labels: {np.bincount(y_test)}")

    # Predict and compute metrics
    logger.info("Predicting on aggregated test set ...")
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "large_confusion_results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")

    # Confusion matrix visualization
    viz = PerformanceVisualizer(outputDir=str(output_dir / "visualizations"))
    viz.plotConfusionMatrix(
        yTrue=y_test,
        yPred=y_pred,
        classNames=["Benign", "Attack"],
        normalize=True,
        savePath=output_dir / "visualizations" / "confusion_matrix_large.png",
    )

    logger.info("Confusion matrix plot generated.")


def main():
    # Choose a large combined dataset (2.9GB+), adjust if needed
    csv_path = "01_Data/02_Processed/combined_user_datasets.csv"
    output_dir = Path("outputs/large_confusion_matrix")

    # Stream through the CSV in chunks to handle 10M+ rows without exhausting memory
    train_and_evaluate_streaming(
        csv_path=csv_path,
        output_dir=output_dir,
        chunk_size=200_000,
        max_test=1_000_000,
    )


if __name__ == "__main__":
    main()
