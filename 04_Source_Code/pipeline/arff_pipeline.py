from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

from services.data_ingestion.arff_loader import find_arff_files_in_dirs, load_multiple_arff
from services.ensemble_service.ensemble_classifier import EnsembleClassifier


def combine_arff_to_dataframe(arff_dirs: List[str]) -> pd.DataFrame:

    files = find_arff_files_in_dirs(arff_dirs)
    if not files:
        raise FileNotFoundError("No .arff files found in provided directories")
    df = load_multiple_arff(files)
    return df


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:

    # Prefer common label names
    candidates = [c for c in df.columns if str(c).lower() in ['class', 'label', 'target', 'attack_cat', 'attack', 'category']]
    if candidates:
        label_col = candidates[0]
    else:
        # Fallback: pick last column
        label_col = df.columns[-1]

    y = df[label_col].astype(str)
    X = df.drop(columns=[label_col])
    return X, y


def _stratified_subsample(X: pd.DataFrame, y: pd.Series, max_samples: int = 20000, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:

    n = len(y)
    if n <= max_samples:
        return X, y
    # compute per-class sample counts
    frac = max_samples / float(n)
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=frac, stratify=y, random_state=random_state
    )
    return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)


def train_evaluate_ensemble(df: pd.DataFrame) -> dict:

    X_df, y = split_features_labels(df)
    # Convert to numeric; coerce errors for any non-numeric, fill NaNs
    X_df = X_df.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Subsample to control memory
    X_df, y = _stratified_subsample(X_df, y, max_samples=20000, random_state=42)
    X = X_df.to_numpy(dtype=np.float32)

    # Drop classes with <2 occurrences; else use non-stratified split
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask_valid = y.isin(valid_classes)
    X = X[mask_valid.values]
    y = y[mask_valid].reset_index(drop=True)

    stratify_arg = y if (y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # Dimensionality reduction to limit memory usage
    n_features = X_train_s.shape[1]
    n_components = min(32, n_features)
    if n_features > n_components:
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
        X_train_s = pca.fit_transform(X_train_s).astype(np.float32)
        X_test_s = pca.transform(X_test_s).astype(np.float32)

    try:
        clf = EnsembleClassifier(
            base_dim=X_train_s.shape[1],
            num_classes=len(np.unique(y_train)),
            n_estimators=50,
            max_depth=12,
            max_features='sqrt',
            n_jobs=1,
            random_state=42,
        )
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        model_used = 'RandomForest-Ensemble'
    except MemoryError:
        # Fallback to memory-efficient linear classifier
        lr = LogisticRegression(
            solver='saga',
            multi_class='multinomial',
            max_iter=200,
            n_jobs=1,
            random_state=42
        )
        lr.fit(X_train_s, y_train)
        preds = lr.predict(X_test_s)
        model_used = 'LogisticRegression-saga'

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    report = classification_report(y_test, preds, output_dict=False)

    return {
        'accuracy': acc,
        'f1_weighted': f1,
        'report_text': report,
        'num_samples': len(df),
        'num_features': X_train_s.shape[1],
        'num_classes': len(np.unique(y)),
        'model_used': model_used
    }


