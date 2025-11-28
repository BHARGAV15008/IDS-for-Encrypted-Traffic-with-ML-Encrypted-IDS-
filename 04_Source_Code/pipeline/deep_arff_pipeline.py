from typing import List, Tuple, Dict
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

from services.data_ingestion.arff_loader import find_arff_files_in_dirs, load_multiple_arff
from services.model_service.hybrid_attention_model import HybridCNNBiLSTMAttention


def combine_arff_numeric(arff_dirs: List[str]) -> Tuple[pd.DataFrame, pd.Series]:

    files = find_arff_files_in_dirs(arff_dirs)
    if not files:
        raise FileNotFoundError("No .arff files found in provided directories")
    df = load_multiple_arff(files)

    # label selection
    candidates = [c for c in df.columns if str(c).lower() in ['class', 'label', 'target', 'attack_cat', 'attack', 'category']]
    label_col = candidates[0] if candidates else df.columns[-1]
    y = df[label_col].astype(str)
    X = df.drop(columns=[label_col])

    # numeric coercion
    X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y


def tabular_to_sequences(X: np.ndarray, seq_len: int = 20) -> Tuple[np.ndarray, int]:

    num_samples, num_feats = X.shape
    feat_dim = math.ceil(num_feats / seq_len)
    padded = feat_dim * seq_len
    if padded != num_feats:
        pad_cols = padded - num_feats
        X = np.pad(X, ((0, 0), (0, pad_cols)), mode='constant', constant_values=0.0)
    X_seq = X.reshape(num_samples, seq_len, feat_dim)
    return X_seq.astype(np.float32), feat_dim


def prepare_torch_loaders(X_df: pd.DataFrame, y: pd.Series, seq_len: int = 32, batch_size: int = 128, test_size: float = 0.2, random_state: int = 42):

    # remove rare classes (<2) then split
    counts = y.value_counts()
    valid = counts[counts >= 2].index
    mask = y.isin(valid)
    X_df = X_df[mask]
    y = y[mask]

    # scale numeric
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_df.values).astype(np.float32)

    # encode labels
    le = LabelEncoder()
    y_np = le.fit_transform(y.values)

    # reshape to sequences
    X_seq, feat_dim = tabular_to_sequences(X_np, seq_len=seq_len)

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_np, test_size=test_size, random_state=random_state, stratify=y_np if len(np.unique(y_np)) > 1 else None)

    # Create validation split from train for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=random_state, stratify=y_train if len(np.unique(y_train)) > 1 else None
    )

    # Weighted sampler to address class imbalance
    class_counts = np.bincount(y_tr)
    class_counts[class_counts == 0] = 1
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_tr]
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(sample_weights), replacement=True)

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long())
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long())
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, feat_dim, len(le.classes_), le, class_counts


def _focal_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = nn.functional.cross_entropy(logits, targets, weight=weights, reduction='none')
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


def train_hybrid_model(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, feat_dim: int, num_classes: int,
                       epochs: int = 15, lr: float = 1e-3, device: str = 'cpu', class_counts: np.ndarray | None = None,
                       use_focal: bool = True, gamma: float = 2.0) -> Dict[str, float]:

    model = HybridCNNBiLSTMAttention(input_features=feat_dim, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    class_weights_t = None
    if class_counts is not None:
        class_counts = class_counts.astype(np.float32)
        class_counts[class_counts == 0] = 1
        inv = 1.0 / class_counts
        inv = inv / inv.sum() * len(inv)
        class_weights_t = torch.tensor(inv, dtype=torch.float32, device=device)

    best_f1 = -1.0
    patience = 3
    bad_epochs = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            if use_focal:
                loss = _focal_loss(logits, yb, class_weights_t if class_weights_t is not None else None, gamma=gamma)
            else:
                loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
                loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.append(preds)
                val_true.append(yb.numpy())
        yv = np.concatenate(val_true)
        pv = np.concatenate(val_preds)
        val_f1 = f1_score(yv, pv, average='weighted') if len(np.unique(yv)) > 1 else 0.0
        scheduler.step(val_f1)
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            bad_epochs = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    # load best
    if 'best_state' in locals():
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # test
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted') if len(np.unique(y_true)) > 1 else 0.0
    report = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
    return {"accuracy": acc, "f1_weighted": f1, "report_text": report}


def run_arff_deep(arff_dirs: List[str], seq_len: int = 32, batch_size: int = 128, epochs: int = 15, device: str = 'cpu', use_focal: bool = True, gamma: float = 2.0) -> Dict[str, float]:

    X_df, y = combine_arff_numeric(arff_dirs)
    train_loader, val_loader, test_loader, feat_dim, num_classes, _, class_counts = prepare_torch_loaders(
        X_df, y, seq_len=seq_len, batch_size=batch_size
    )
    metrics = train_hybrid_model(train_loader, val_loader, test_loader, feat_dim, num_classes, epochs=epochs, device=device, class_counts=class_counts, use_focal=use_focal, gamma=gamma)
    metrics.update({
        "seq_len": seq_len,
        "feat_dim": feat_dim,
        "num_classes": num_classes
    })
    return metrics


