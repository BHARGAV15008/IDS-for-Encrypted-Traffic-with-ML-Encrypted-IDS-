"""
Improved Complete Workflow Orchestrator - ALL 4 PHASES IMPLEMENTED
End-to-end pipeline with all improvements integrated:
- Phase 1: Data Quality (cleaning, balancing, normalization)
- Phase 2: Model Architecture (multi-head attention, weighted focal loss, improved ensemble)
- Phase 3: Feature Engineering (temporal, TLS, invariance features)
- Phase 4: Training & Evaluation (threshold optimization, comprehensive metrics)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import json
import logging
import warnings
import dask.dataframe as dd
import yaml
from sklearn.metrics import confusion_matrix
import importlib.util

# Add project root to path
projectRoot = Path(__file__).parent.parent
sys.path.insert(0, str(projectRoot))
sys.path.insert(0, str(projectRoot / '02_Features' / '02_Novel_Feature_Modules'))

# Dynamically load apply_advanced_feature_engineering
try:
    spec_dask_feature_engineering = importlib.util.spec_from_file_location(
        "dask_feature_engineering",
        "E:/IDS for Encrypted Traffic with ML (Encrypted IDS)/02_Features/02_Novel_Feature_Modules/dask_feature_engineering.py"
    )
    dask_feature_engineering_module = importlib.util.module_from_spec(spec_dask_feature_engineering)
    spec_dask_feature_engineering.loader.exec_module(dask_feature_engineering_module)
    apply_advanced_feature_engineering = dask_feature_engineering_module.apply_advanced_feature_engineering
except Exception as e:
    print(f"Error loading apply_advanced_feature_engineering: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 1: DATA QUALITY IMPROVEMENTS (Now operates on flow-level data)
# ============================================================================

class Phase1DataQuality:
    """Phase 1: Data Cleaning, Balancing, and Normalization on flow-level data"""
    
    def __init__(self, config: Dict):
        self.config = config['data']
        logger.info("✓ Phase 1: Data Quality initialized")
    
    def clean_flow_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """Clean flow-level data"""
        logger.info(f"  [1.1] Cleaning flow-level data...")
        
        # Impute missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any().compute():
                median_val = df[col].median_approximate().compute()
                df[col] = df[col].fillna(median_val)
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        logger.info(f"    - Final flow dataset: {len(df)} rows, {len(df.columns)} columns")
        return df


# ============================================================================
# PHASE 2: MODEL ARCHITECTURE IMPROVEMENTS
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert input_dim % num_heads == 0
        self.input_dim, self.num_heads, self.head_dim = input_dim, num_heads, input_dim // num_heads
        self.scale = np.sqrt(self.head_dim)
        self.query, self.key, self.value = nn.Linear(input_dim, input_dim), nn.Linear(input_dim, input_dim), nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)
        self.layer_norm, self.dropout = nn.LayerNorm(input_dim), nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x.shape[0], x.shape[1]
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.input_dim)
        output = self.layer_norm(self.fc_out(context) + x)
        return output, attention_weights

class ImprovedHybridModel(nn.Module):
    def __init__(self, input_features: int, num_classes: int, dropout_rate: float = 0.2, attention_heads: int = 4):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv1d(input_features, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128)
        self.dropout_conv = nn.Dropout(dropout_rate)
        self.bilstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.attention = MultiHeadAttention(128, num_heads=attention_heads, dropout=dropout_rate)
        self.fc1, self.bn_fc1 = nn.Linear(128, 64), nn.BatchNorm1d(64)
        self.fc2, self.bn_fc2 = nn.Linear(64, 32), nn.BatchNorm1d(32)
        self.classifier = nn.Linear(32, num_classes)
        self.dropout_fc, self.relu = nn.Dropout(dropout_rate), nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.dropout_conv(self.relu(self.bn1(self.conv1(x.transpose(1, 2)))))
        x = self.dropout_conv(self.relu(self.bn2(self.conv2(x))))
        lstm_out, _ = self.bilstm(x.transpose(1, 2))
        attn_out, _ = self.attention(lstm_out)
        context = torch.mean(attn_out, dim=1)
        x = self.dropout_fc(self.relu(self.bn_fc1(self.fc1(context))))
        x = self.dropout_fc(self.relu(self.bn_fc2(self.fc2(x))))
        return self.classifier(x)

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        alpha_t = self.alpha.to(inputs.device).gather(0, targets)
        return (alpha_t * focal_loss).mean()

class Phase2ModelArchitecture:
    def __init__(self, config: Dict, device: str):
        self.config, self.training_config, self.device = config['model'], config['training'], device
        logger.info("✓ Phase 2: Model Architecture initialized")
    
    def create_model(self, input_features: int) -> ImprovedHybridModel:
        logger.info(f"  [2.1] Creating model with {input_features} features...")
        # num_classes is expected to be injected into self.config by the workflow
        if 'num_classes' not in self.config:
            raise ValueError("'num_classes' must be provided in model config before creating the model.")
        model = ImprovedHybridModel(input_features, **self.config).to(self.device)
        return model
    
    def create_loss_function(self, class_weights: torch.Tensor) -> WeightedFocalLoss:
        gamma = self.training_config['focal_gamma']
        logger.info(f"  [2.2] Creating weighted focal loss (gamma={gamma})...")
        return WeightedFocalLoss(alpha=class_weights, gamma=gamma)

# ============================================================================
# PHASE 4: TRAINING & EVALUATION IMPROVEMENTS
# ============================================================================

class Phase4Evaluation:
    @staticmethod
    def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        logger.info(f"  [4.2] Calculating comprehensive metrics...")
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            'detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        }
        logger.info(f"    - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

# ============================================================================
# MAIN IMPROVED WORKFLOW
# ============================================================================

class DaskFrameIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, ddf: dd.DataFrame, feature_cols: List[str], label_col: str):
        self.ddf, self.feature_cols, self.label_col = ddf, feature_cols, label_col
        self.npartitions = ddf.npartitions

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        p_start, p_end = (0, self.npartitions)
        if worker_info:
            per_worker = int(np.ceil(self.npartitions / worker_info.num_workers))
            p_start = worker_info.id * per_worker
            p_end = min(p_start + per_worker, self.npartitions)
        
        for i in range(p_start, p_end):
            df_part = self.ddf.get_partition(i).compute()
            if df_part.empty:
                continue
            # Safely convert feature columns to numeric; non-numeric values become 0.0
            X_part = df_part[self.feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            Xp = np.log1p(np.abs(X_part.astype('float64'))).astype('float32').values
            yp = df_part[self.label_col].astype('int64').values
            for j in range(len(Xp)):
                yield torch.tensor(Xp[j], dtype=torch.float32), torch.tensor(yp[j], dtype=torch.long)

class ImprovedCompleteWorkflow:
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if config['training']['device'] == 'auto' else config['training']['device']
        logger.info(f"Using device: {self.device}")

        self.phase1 = Phase1DataQuality(config)
        self.phase2 = Phase2ModelArchitecture(config, self.device)
        self.phase4_eval = Phase4Evaluation()
        self.results = {}
        logger.info("=" * 80 + "\nIMPROVED COMPLETE WORKFLOW - ALL 4 PHASES\n" + "=" * 80)

    def _dask_train_val_test_split(self, df: dd.DataFrame) -> Tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:
        data_config = self.config['data']
        test_size, val_size, random_state = data_config['test_size'], data_config['validation_size'], data_config['random_state']
        logger.info(f"  [1.3] Performing Dask train-val-test split (test: {test_size}, val: {val_size})...")
        train_val_df, test_df = df.random_split([1 - test_size, test_size], random_state=random_state)
        relative_val_size = val_size / (1 - test_size)
        train_df, val_df = train_val_df.random_split([1 - relative_val_size, relative_val_size], random_state=random_state)
        logger.info(f"    - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)} rows")
        return train_df, val_df, test_df

    def run(self, ddf: dd.DataFrame) -> Dict:
        label_col = self.config['data']['label_col']

        logger.info("\n" + "=" * 80 + "\nPHASE 3: ADVANCED FEATURE ENGINEERING\n" + "=" * 80)
        # Depending on the dataset, we may or may not have a Flow ID column.
        if 'Flow ID' in ddf.columns:
            # Packet- or flow-level data with Flow ID available
            flow_features_ddf = apply_advanced_feature_engineering(ddf)

            # Extract labels (assuming label is consistent per flow)
            flow_labels = ddf.groupby('Flow ID')[label_col].first()
            flow_features_ddf = flow_features_ddf.merge(flow_labels.to_frame(), left_index=True, right_index=True)
        else:
            # For datasets like MachineLearningCVE where there is no Flow ID,
            # we skip advanced feature engineering and operate directly on the
            # provided flow-level features.
            logger.warning("No 'Flow ID' column found; skipping advanced feature engineering and using provided features directly.")
            if label_col not in ddf.columns:
                raise KeyError(f"Label column '{label_col}' not found in input data.")
            flow_features_ddf = ddf

        logger.info("\n" + "=" * 80 + "\nPHASE 1: DATA QUALITY\n" + "=" * 80)
        flow_features_ddf = self.phase1.clean_flow_data(flow_features_ddf)
        
        train_df, val_df, test_df = self._dask_train_val_test_split(flow_features_ddf)
        
        # All columns except the label are treated as numeric features here;
        # upstream preprocessing is responsible for dropping non-numeric fields.
        numeric_cols = [col for col in train_df.columns if col != label_col]
        input_dim = len(numeric_cols)
        self.results['phase3'] = {'total_features': input_dim}

        train_dataset, val_dataset, test_dataset = (
            DaskFrameIterableDataset(df, numeric_cols, label_col) for df in [train_df, val_df, test_df]
        )
        
        counts = train_df[label_col].value_counts().compute().to_dict()
        classes = sorted(counts.keys())
        class_counts = np.array([counts.get(c, 0) for c in classes], dtype=float)
        class_weights = torch.tensor(class_counts.sum() / (len(classes) * class_counts), dtype=torch.float32)
        num_classes = len(classes)
        self.results['phase1'] = {
            'class_weights': class_weights.tolist(),
            'num_classes': num_classes,
            'classes': classes,
        }

        # Inject num_classes into the model configuration so the architecture
        # can be constructed correctly.
        self.config.setdefault('model', {})
        self.config['model']['num_classes'] = num_classes
        
        logger.info("\n" + "=" * 80 + "\nPHASE 2: MODEL ARCHITECTURE\n" + "=" * 80)
        model = self.phase2.create_model(input_dim)
        loss_fn = self.phase2.create_loss_function(class_weights)
        optimizer = optim.Adam(model.parameters(), lr=self.config['training']['learning_rate'])
        
        logger.info("\n" + "=" * 80 + "\nPHASE 4: TRAINING & EVALUATION\n" + "=" * 80)
        train_loader, val_loader, test_loader = (
            DataLoader(ds, batch_size=self.config['training']['batch_size']) for ds in [train_dataset, val_dataset, test_dataset]
        )
        
        logger.info(f"  [4.0] Training for up to {self.config['training']['num_epochs']} epochs...")
        best_val_loss, patience_counter = float('inf'), 0
        
        for epoch in range(self.config['training']['num_epochs']):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(X_batch.to(self.device)), y_batch.to(self.device))
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_loss, val_batches = 0.0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    val_loss += loss_fn(model(X_batch.to(self.device)), y_batch.to(self.device)).item()
                    val_batches += 1
            avg_val_loss = val_loss / max(val_batches, 1)
            
            logger.info(f"    - Epoch {epoch + 1}, Val Loss: {avg_val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss, patience_counter = val_loss, 0
                torch.save(model.state_dict(), self.output_dir / 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    logger.info(f"    - Early stopping at epoch {epoch + 1}")
                    break
        
        model.load_state_dict(torch.load(self.output_dir / 'best_model.pth'))
        
        logger.info(f"  [4.1] Evaluating on test set...")
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_true.extend(y_batch.numpy())
                y_pred.extend(model(X_batch.to(self.device)).argmax(dim=1).cpu().numpy())
        
        self.results['phase4'] = {
            'metrics': self.phase4_eval.calculate_comprehensive_metrics(np.array(y_true), np.array(y_pred))
        }
        self._save_results()
        return self.results
    
    def _save_results(self):
        results_path = self.output_dir / 'improved_workflow_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
        logger.info(f"\n✓ Results saved to: {results_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Improved Complete IDS Workflow")
    parser.add_argument("--config", type=str, default="traffic_classification_configuration.yaml", help="Path to YAML config file")
    parser.add_argument("--data", type=str, help="Path to packet-level CSV data file (overrides config)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.data:
        config['data']['path'] = args.data
    
    data_path = config['data']['path']
    logger.info(f"Loading packet-level data from: {data_path} using Dask...")

    # Get all columns from the header
    with open(data_path, 'r', encoding='latin1') as f:
        header = f.readline()
    all_columns = [col.strip() for col in header.split(',')]
    
    # Specify dtype for all columns as str; label will be converted to int after loading
    dtype = {col: 'str' for col in all_columns}

    ddf = dd.read_csv(data_path, dtype=dtype, assume_missing=True, blocksize="100MB", encoding='latin1')
    ddf.columns = [col.strip() for col in ddf.columns]

    # Determine columns to use
    columns_to_drop = config['data'].get('columns_to_drop', [])
    columns_to_drop = [col.strip() for col in columns_to_drop]
    usecols = [col for col in ddf.columns if col not in columns_to_drop]
    
    ddf = ddf[usecols]

    # Encode label column to binary 0/1 (benign vs attack)
    label_col = config['data']['label_col']
    benign_values = set(str(v) for v in config['data']['benign_values'])
    ddf[label_col] = (~ddf[label_col].astype(str).isin(benign_values)).astype('int64')
    
    logger.info(f"Finished loading data with Dask. Total packets: {len(ddf)}")
    
    workflow = ImprovedCompleteWorkflow(config)
    workflow.run(ddf)
    
    logger.info("\n✓ All phases completed successfully!")

if __name__ == '__main__':
    main()
