from typing import List, Tuple, Dict
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from services.data_ingestion.arff_loader import find_arff_files_in_dirs, load_arff_file


def _infer_encryption_flag_from_path(p: str) -> int:

    low = p.lower()
    if 'no-vpn' in low or 'no_vpn' in low or 'novpn' in low:
        return 0
    if 'vpn' in low:
        return 1
    # Scenario A1 is VPN (encrypted)
    if 'scenario a1' in low or 'scenario_a1' in low:
        return 1
    # Unknown -> -1 (will drop)
    return -1


def build_annotated_frame(arff_dirs: List[str]) -> pd.DataFrame:

    files: List[str] = []
    for d in arff_dirs:
        files.extend(find_arff_files_in_dirs([d]))
    frames: List[pd.DataFrame] = []
    for f in files:
        df = load_arff_file(f)
        df['__source_file'] = f
        df['__enc_flag'] = _infer_encryption_flag_from_path(f)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    return full


def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str]:

    candidates = [c for c in df.columns if str(c).lower() in ['class', 'label', 'target', 'attack_cat', 'attack', 'category']]
    label_col = candidates[0] if candidates else df.columns[-1]
    y = df[label_col].astype(str)
    X = df.drop(columns=[label_col])
    return X, y, label_col


def train_binary_encryption_classifier(df: pd.DataFrame) -> Dict:

    df = df[df['__enc_flag'] >= 0].copy()
    X_all = df.drop(columns=['__enc_flag', '__source_file'])
    X_all = X_all.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_bin = df['__enc_flag'].astype(int).values

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_bin, test_size=0.2, random_state=42, stratify=y_bin if len(np.unique(y_bin))>1 else None)

    clf = LogisticRegression(max_iter=200, n_jobs=1, solver='lbfgs')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    return {
        'scaler': scaler,
        'clf': clf,
        'metrics': {
            'accuracy': float(accuracy_score(y_test, preds)),
            'f1_weighted': float(f1_score(y_test, preds, average='weighted')),
            'report_text': classification_report(y_test, preds, output_dict=False)
        }
    }


def train_intrusion_on_encrypted(df: pd.DataFrame) -> Dict:

    df = df[df['__enc_flag'] == 1].copy()
    X_df, y, label_col = split_X_y(df)

    X_df = X_df.drop(columns=['__enc_flag', '__source_file'], errors='ignore')
    X_df = X_df.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)

    le = LabelEncoder()
    y_enc = le.fit_transform(y.values)

    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_df.values).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X_np, y_enc, test_size=0.2, random_state=42, stratify=y_enc if len(np.unique(y_enc))>1 else None)

    clf = RandomForestClassifier(n_estimators=200, max_depth=20, max_features='sqrt', n_jobs=1, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    return {
        'scaler': scaler,
        'label_encoder': le,
        'clf': clf,
        'metrics': {
            'accuracy': float(accuracy_score(y_test, preds)),
            'f1_weighted': float(f1_score(y_test, preds, average='weighted')),
            'report_text': classification_report(y_test, preds, output_dict=False)
        }
    }


def run_two_stage(arff_dirs: List[str]) -> Dict:

    full = build_annotated_frame(arff_dirs)
    stage1 = train_binary_encryption_classifier(full)
    stage2 = train_intrusion_on_encrypted(full)

    return {
        'stage1_encryption_metrics': stage1['metrics'],
        'stage2_encrypted_intrusion_metrics': stage2['metrics']
    }


