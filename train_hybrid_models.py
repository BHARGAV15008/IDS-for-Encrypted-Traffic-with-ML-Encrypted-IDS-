"""
Script to train Hybrid CNN-LSTM and Hybrid CNN-BiLSTM-Attention models
on the CIC-IDS2017 dataset.
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import importlib.util

# Dynamically load DataIngestionService
try:
    spec_data_ingestion = importlib.util.spec_from_file_location(
        "dataIngestionService",
        "e:/IDS for Encrypted Traffic with ML (Encrypted IDS)/04_Source_Code/services/dataIngestionService.py"
    )
    dataIngestionService_module = importlib.util.module_from_spec(spec_data_ingestion)
    spec_data_ingestion.loader.exec_module(dataIngestionService_module)
    DataIngestionService = dataIngestionService_module.DataIngestionService
except Exception as e:
    print(f"Error loading DataIngestionService: {e}")
    sys.exit(1)

# Dynamically load ModelTrainingService
try:
    spec_model_training = importlib.util.spec_from_file_location(
        "modelTrainingService",
        "e:/IDS for Encrypted Traffic with ML (Encrypted IDS)/04_Source_Code/services/modelTrainingService.py"
    )
    modelTrainingService_module = importlib.util.module_from_spec(spec_model_training)
    spec_model_training.loader.exec_module(modelTrainingService_module)
    ModelTrainingService = modelTrainingService_module.ModelTrainingService
except Exception as e:
    print(f"Error loading ModelTrainingService: {e}")
    sys.exit(1)

# Dynamically load HybridCNNLSTM
try:
    spec_hybrid_cnn_lstm = importlib.util.spec_from_file_location(
        "hybrid_cnn_lstm",
        "e:/IDS for Encrypted Traffic with ML (Encrypted IDS)/03_Models/01_Architectures/hybrid_cnn_lstm.py"
    )
    hybrid_cnn_lstm_module = importlib.util.module_from_spec(spec_hybrid_cnn_lstm)
    spec_hybrid_cnn_lstm.loader.exec_module(hybrid_cnn_lstm_module)
    HybridCNNLSTM = hybrid_cnn_lstm_module.HybridCNNLSTM
except Exception as e:
    print(f"Error loading HybridCNNLSTM: {e}")
    sys.exit(1)

# Dynamically load HybridCnnBiLstmAttention
try:
    spec_hybrid_cnn_bilstm_attention = importlib.util.spec_from_file_location(
        "hybridCnnBiLstmAttention",
        "e:/IDS for Encrypted Traffic with ML (Encrypted IDS)/03_Models/01_Architectures/hybridCnnBiLstmAttention.py"
    )
    hybrid_cnn_bilstm_attention_module = importlib.util.module_from_spec(spec_hybrid_cnn_bilstm_attention)
    spec_hybrid_cnn_bilstm_attention.loader.exec_module(hybrid_cnn_bilstm_attention_module)
    HybridCnnBiLstmAttention = hybrid_cnn_bilstm_attention_module.HybridCnnBiLstmAttention
except Exception as e:
    print(f"Error loading HybridCnnBiLstmAttention: {e}")
    sys.exit(1)

print("Successfully loaded services and architectures dynamically.")

class TimeSeriesDataset(Dataset):
    """Dataset class for time series data for hybrid models."""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Initialize time series dataset.
        
        Args:
            sequences: Array of shape (num_sequences, window_size, num_features)
            labels: Array of shape (num_sequences,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]

def load_time_series_data():
    """Load preprocessed time series data for hybrid models."""
    
    print("Loading preprocessed time series data...")
    
    # Load the time series dataset (window size 10)
    time_series_file = 'e:/IDS for Encrypted Traffic with ML (Encrypted IDS)/01_Data/03_TimeSeries/timeseries_window10.npz'
    
    try:
        data = np.load(time_series_file)
        sequences = data['sequences']
        labels = data['labels']
        
        print(f"Time Series Data loaded successfully!")
        print(f"Sequences shape: {sequences.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Window size: {data['window_size']}")
        print(f"Number of features: {data['num_features']}")
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"Classes: {np.unique(labels)}")

        # Remap labels if -1 is present
        if -1 in labels:
            print("Remapping label -1 to 2...")
            labels[labels == -1] = 2
            print(f"Updated Classes: {np.unique(labels)}")

        # Ensure labels are non-negative
        if np.min(labels) < 0:
            raise ValueError(f"Negative labels found in dataset: {np.unique(labels)}")

        return sequences, labels
        
    except Exception as e:
        print(f"Error loading time series data: {e}")
        raise

def train_hybrid_models():
    """Train both hybrid models on the time series dataset."""
    
    # Load preprocessed time series data
    sequences, labels = load_time_series_data()
    
    print(f"\nDataset Summary:")
    print(f"Sequences shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"Classes: {np.unique(labels)}")
    
    # Initialize model training service
    print("Initializing Model Training Service...")
    mts = ModelTrainingService(
        hybrid_cnn_lstm_class=HybridCNNLSTM,
        hybrid_cnn_bilstm_attention_class=HybridCnnBiLstmAttention
    )
    
    # Override the default dataset creation to use our TimeSeriesDataset
    def train_with_time_series_dataset(sequences, labels, modelType, modelConfig, validationSplit=0.2):
        """Custom training function that uses TimeSeriesDataset for hybrid models."""
        print(f"[train_with_time_series_dataset] Starting custom training function for modelType: {modelType}")
        
        # Create time series dataset
        print("[train_with_time_series_dataset] Creating TimeSeriesDataset...")
        dataset = TimeSeriesDataset(sequences, labels)
        print("[train_with_time_series_dataset] TimeSeriesDataset created.")
        
        # Split into train and validation
        print("[train_with_time_series_dataset] Splitting data into train and validation...")
        datasetSize = len(dataset)
        indices = list(range(datasetSize))
        split = int(np.floor(validationSplit * datasetSize))
        np.random.shuffle(indices)
        
        trainIndices, valIndices = indices[split:], indices[:split]
        print("[train_with_time_series_dataset] Data split complete. Creating DataLoaders...")
        
        trainLoader = DataLoader(
            dataset,
            batch_size=mts.batchSize,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(trainIndices)
        )
        valLoader = DataLoader(
            dataset,
            batch_size=mts.batchSize,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(valIndices)
        )
        print("[train_with_time_series_dataset] DataLoaders created.")
        
        # Create model
        print(f"[train_with_time_series_dataset] Creating model of type: {modelType}...")
        model = mts._createModel(modelType, modelConfig)
        model = model.to(mts.device)
        print(f"[train_with_time_series_dataset] Model {modelType} created and moved to device: {mts.device}.")
        
        # Setup training
        print("[train_with_time_series_dataset] Setting up training components (criterion, optimizer, scheduler)...")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=mts.learningRate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        print("[train_with_time_series_dataset] Training components setup complete.")
        
        # Training loop
        bestValAcc = 0.0
        patienceCounter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        print(f"[train_with_time_series_dataset] Starting training loop for {mts.numEpochs} epochs.")
        
        for epoch in range(mts.numEpochs):
            # Training phase
            trainLoss, trainAcc = mts._trainEpoch(model, trainLoader, criterion, optimizer, epoch)
            
            # Validation phase
            valLoss, valAcc = mts._validateEpoch(model, valLoader, criterion, epoch)
            
            # Update learning rate
            scheduler.step(valAcc)
            
            # Record history
            history['train_loss'].append(trainLoss)
            history['train_acc'].append(trainAcc)
            history['val_loss'].append(valLoss)
            history['val_acc'].append(valAcc)
            
            print(
                f"Epoch {epoch+1}/{mts.numEpochs} - "
                f"Train Loss: {trainLoss:.4f}, Train Acc: {trainAcc:.4f}, "
                f"Val Loss: {valLoss:.4f}, Val Acc: {valAcc:.4f}"
            )
            
            # Early stopping and checkpointing
            if valAcc > bestValAcc:
                bestValAcc = valAcc
                patienceCounter = 0
                
                # Save best model
                checkpointPath = mts.outputDir / f"best_{modelType}_model.pth"
                mts._saveCheckpoint(model, optimizer, epoch, valAcc, checkpointPath)
                print(f"Saved best model with val_acc: {valAcc:.4f}")
            else:
                patienceCounter += 1
                
                if patienceCounter >= mts.earlyStoppingPatience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model
        mts._loadCheckpoint(model, mts.outputDir / f"best_{modelType}_model.pth")
        
        # Final evaluation
        finalMetrics = mts._evaluateModel(model, valLoader)
        
        return {
            'modelType': modelType,
            'bestValAccuracy': bestValAcc,
            'finalMetrics': finalMetrics,
            'trainingHistory': history,
            'checkpointPath': str(mts.outputDir / f"best_{modelType}_model.pth")
        }
    
    # Common configuration
    num_classes = len(np.unique(labels))
    window_size = sequences.shape[1]  # 10
    num_features = sequences.shape[2]  # 26
    
    # Hybrid CNN-LSTM model configuration
    hybrid_cnn_lstm_config = {
        'packet_input_size': num_features,  # Number of features per time step
        'num_classes': num_classes,
        'cnn_channels': [64, 128],
        'cnn_kernels': [3, 5],
        'lstm_hidden_size': 128,
        'lstm_layers': 2,
        'dropout_rate': 0.3,
        'use_attention': True,
        'use_residual': True
    }
    
    # Hybrid CNN-BiLSTM-Attention model configuration
    hybrid_cnn_bilstm_attention_config = {
        'inputFeatures': num_features,  # Number of features per time step
        'numClasses': num_classes,
        'cnnChannels': [64, 128, 256],
        'lstmHiddenSize': 128,
        'lstmNumLayers': 2,
        'attentionHeads': 8,
        'dropout': 0.3,
        'useCnnResidual': True,
        'useFeatureAttention': True,
        'useTemporalAttention': True
    }
    
    # Train Hybrid CNN-LSTM model
    print("\n" + "="*60)
    print("TRAINING HYBRID CNN-LSTM MODEL")
    print("="*60)
    
    try:
        training_result_cnn_lstm = train_with_time_series_dataset(
            sequences=sequences,
            labels=labels,
            modelType='hybrid_cnn_lstm',
            modelConfig=hybrid_cnn_lstm_config,
            validationSplit=0.2
        )
        
        print(f"Hybrid CNN-LSTM Training completed!")
        print(f"Best validation accuracy: {training_result_cnn_lstm['bestValAccuracy']:.4f}")
        print(f"Final metrics: {training_result_cnn_lstm['finalMetrics']}")
        
    except Exception as e:
        print(f"Error training Hybrid CNN-LSTM model: {e}")
        import traceback
        traceback.print_exc()
    
    # Train Hybrid CNN-BiLSTM-Attention model
    print("\n" + "="*60)
    print("TRAINING HYBRID CNN-BiLSTM-ATTENTION MODEL")
    print("="*60)
    
    try:
        training_result_bilstm_attention = train_with_time_series_dataset(
            sequences=sequences,
            labels=labels,
            modelType='hybrid_cnn_bilstm_attention',
            modelConfig=hybrid_cnn_bilstm_attention_config,
            validationSplit=0.2
        )
        
        print(f"Hybrid CNN-BiLSTM-Attention Training completed!")
        print(f"Best validation accuracy: {training_result_bilstm_attention['bestValAccuracy']:.4f}")
        print(f"Final metrics: {training_result_bilstm_attention['finalMetrics']}")
        
    except Exception as e:
        print(f"Error training Hybrid CNN-BiLSTM-Attention model: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        'cnn_lstm_result': training_result_cnn_lstm if 'training_result_cnn_lstm' in locals() else None,
        'bilstm_attention_result': training_result_bilstm_attention if 'training_result_bilstm_attention' in locals() else None
    }

if __name__ == "__main__":
    print("Starting Hybrid Model Training Script")
    print("="*60)
    
    results = train_hybrid_models()
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if results['cnn_lstm_result']:
        print("Hybrid CNN-LSTM Results:")
        print(f"  Best Validation Accuracy: {results['cnn_lstm_result']['bestValAccuracy']:.4f}")
        print(f"  Final Accuracy: {results['cnn_lstm_result']['finalMetrics']['accuracy']:.4f}")
        print(f"  F1 Score: {results['cnn_lstm_result']['finalMetrics']['f1Score']:.4f}")
    
    if results['bilstm_attention_result']:
        print("Hybrid CNN-BiLSTM-Attention Results:")
        print(f"  Best Validation Accuracy: {results['bilstm_attention_result']['bestValAccuracy']:.4f}")
        print(f"  Final Accuracy: {results['bilstm_attention_result']['finalMetrics']['accuracy']:.4f}")
        print(f"  F1 Score: {results['bilstm_attention_result']['finalMetrics']['f1Score']:.4f}")
    
    print("\nTraining script completed!")