"""
Model Training Service with Hyperparameter Tuning and Cross-Validation

Advanced training capabilities including:
- K-Fold Cross Validation
- Hyperparameter tuning with Optuna
- Early stopping
- Learning rate scheduling
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
import json
import logging
from datetime import datetime
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import optuna
import sys
import os
import importlib.util

# Dynamically load BaseService
try:
    spec_base_service = importlib.util.spec_from_file_location(
        "baseService",
        "E:/IDS for Encrypted Traffic with ML (Encrypted IDS)/04_Source_Code/services/baseService.py"
    )
    baseService_module = importlib.util.module_from_spec(spec_base_service)
    spec_base_service.loader.exec_module(baseService_module)
    BaseService = baseService_module.BaseService
except Exception as e:
    print(f"Error loading BaseService: {e}")
    sys.exit(1)
# Dynamically load BiLSTMModel (aliasing LSTMFeatureExtractor)
try:
    spec_lstm_model = importlib.util.spec_from_file_location(
        "lstmModule",
        "E:/IDS for Encrypted Traffic with ML (Encrypted IDS)/03_Models/01_Architectures/lstmModule.py"
    )
    lstmModule_module = importlib.util.module_from_spec(spec_lstm_model)
    spec_lstm_model.loader.exec_module(lstmModule_module)
    BiLSTMModel = lstmModule_module.LSTMFeatureExtractor # Assuming LSTMFeatureExtractor is the intended BiLSTMModel
except Exception as e:
    print(f"Error loading BiLSTMModel (LSTMFeatureExtractor from lstmModule): {e}")
    sys.exit(1)
# Dynamically load CNNModel (aliasing CNNFeatureExtractor)
try:
    spec_cnn_model = importlib.util.spec_from_file_location(
        "cnnModule",
        "E:/IDS for Encrypted Traffic with ML (Encrypted IDS)/03_Models/01_Architectures/cnnModule.py"
    )
    cnnModule_module = importlib.util.module_from_spec(spec_cnn_model)
    spec_cnn_model.loader.exec_module(cnnModule_module)
    CNNModel = cnnModule_module.CNNFeatureExtractor # Assuming CNNFeatureExtractor is the intended CNNModel
except Exception as e:
    print(f"Error loading CNNModel (CNNFeatureExtractor from cnnModule): {e}")
    sys.exit(1)
# Dynamically load FocalLoss
try:
    spec_focal_loss = importlib.util.spec_from_file_location(
        "customLosses",
        "E:/IDS for Encrypted Traffic with ML (Encrypted IDS)/03_Models/01_Architectures/customLosses.py"
    )
    customLosses_module = importlib.util.module_from_spec(spec_focal_loss)
    spec_focal_loss.loader.exec_module(customLosses_module)
    FocalLoss = customLosses_module.FocalLoss
except Exception as e:
    print(f"Error loading FocalLoss: {e}")
    sys.exit(1)


class NetworkFlowDataset(Dataset):
    """Dataset class for network flow data."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            features: Feature array
            labels: Label array
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class ModelTrainingService(BaseService):
    """Advanced model training service with hyperparameter optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 hybrid_cnn_lstm_class: Any = None,
                 hybrid_cnn_bilstm_attention_class: Any = None):
        """
        Initialize Model Training Service.
        
        Args:
            config: Training configuration
            hybrid_cnn_lstm_class: The HybridCNNLSTM model class
            hybrid_cnn_bilstm_attention_class: The HybridCnnBiLstmAttention model class
        """
        super().__init__("ModelTrainingService", config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.outputDir = Path(self.getConfig('outputDir', './outputs/training'))
        self.outputDir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.numEpochs = self.getConfig('numEpochs', 100)
        self.batchSize = self.getConfig('batchSize', 32)
        self.learningRate = self.getConfig('learningRate', 0.001)
        self.earlyStoppingPatience = self.getConfig('earlyStoppingPatience', 10)
        self.numFolds = self.getConfig('numFolds', 5)
        self.currentEpoch = 0 # Initialize currentEpoch
        self.loss_config = self.getConfig('loss', {'name': 'cross_entropy'})

        self.HybridCNNLSTM = hybrid_cnn_lstm_class
        self.HybridCnnBiLstmAttention = hybrid_cnn_bilstm_attention_class
        
        # Best model tracking
        self.bestModel = None
        self.bestScore = 0.0
        self.trainingHistory = []
    
    def _onStart(self) -> None:
        """Initialize training service."""
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Cross-validation folds: {self.numFolds}")
    
    def _onStop(self) -> None:
        """Cleanup on service stop."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process training request.
        
        Args:
            data: Dictionary containing:
                - action: 'train', 'tune', or 'cross_validate'
                - features: Training features
                - labels: Training labels
                - modelType: 'lstm' or 'cnn'
                - config: Model configuration
                
        Returns:
            Training results
        """
        action = data.get('action', 'train')
        
        if action == 'train':
            return self.trainModel(
                features=data['features'],
                labels=data['labels'],
                modelType=data['modelType'],
                modelConfig=data.get('config', {})
            )
        elif action == 'tune':
            return self.tuneHyperparameters(
                features=data['features'],
                labels=data['labels'],
                modelType=data['modelType'],
                nTrials=data.get('nTrials', 50)
            )
        elif action == 'cross_validate':
            return self.crossValidate(
                features=data['features'],
                labels=data['labels'],
                modelType=data['modelType'],
                modelConfig=data.get('config', {})
            )
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def trainModel(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        modelType: str,
        modelConfig: Dict[str, Any],
        validationSplit: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train a model with early stopping and checkpointing.
        
        Args:
            features: Training features
            labels: Training labels
            modelType: Type of model ('lstm' or 'cnn')
            modelConfig: Model configuration
            validationSplit: Validation split ratio
            
        Returns:
            Training results
        """
        self.logger.info(f"Training {modelType} model...")
        
        # Create dataset
        dataset = NetworkFlowDataset(features, labels)
        
        # Split into train and validation
        datasetSize = len(dataset)
        indices = list(range(datasetSize))
        split = int(np.floor(validationSplit * datasetSize))
        np.random.shuffle(indices)
        
        trainIndices, valIndices = indices[split:], indices[:split]
        
        trainLoader = DataLoader(
            dataset,
            batch_size=self.batchSize,
            sampler=SubsetRandomSampler(trainIndices)
        )
        valLoader = DataLoader(
            dataset,
            batch_size=self.batchSize,
            sampler=SubsetRandomSampler(valIndices)
        )
        
        # Create model
        model = self._createModel(modelType, modelConfig)
        model = model.to(self.device)
        
        # Setup training
        criterion = self._create_loss_function(self.loss_config, trainLoader)
        optimizer = optim.Adam(model.parameters(), lr=self.learningRate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Training loop
        bestValAcc = 0.0
        patienceCounter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(self.numEpochs):
            # Training phase
            trainLoss, trainAcc = self._trainEpoch(model, trainLoader, criterion, optimizer)
            
            # Validation phase
            valLoss, valAcc = self._validateEpoch(model, valLoader, criterion)
            
            # Update learning rate
            scheduler.step(valAcc)
            
            # Record history
            history['train_loss'].append(trainLoss)
            history['train_acc'].append(trainAcc)
            history['val_loss'].append(valLoss)
            history['val_acc'].append(valAcc)
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.numEpochs} - "
                f"Train Loss: {trainLoss:.4f}, Train Acc: {trainAcc:.4f}, "
                f"Val Loss: {valLoss:.4f}, Val Acc: {valAcc:.4f}"
            )
            
            # Early stopping and checkpointing
            if valAcc > bestValAcc:
                bestValAcc = valAcc
                patienceCounter = 0
                
                # Save best model
                checkpointPath = self.outputDir / f"best_{modelType}_model.pth"
                self._saveCheckpoint(model, optimizer, epoch, valAcc, checkpointPath)
                self.logger.info(f"Saved best model with val_acc: {valAcc:.4f}")
            else:
                patienceCounter += 1
                
                if patienceCounter >= self.earlyStoppingPatience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model
        self._loadCheckpoint(model, self.outputDir / f"best_{modelType}_model.pth")
        
        # Final evaluation
        finalMetrics = self._evaluateModel(model, valLoader)
        
        # Save training history
        self._saveTrainingHistory(history, modelType)
        
        return {
            'modelType': modelType,
            'bestValAccuracy': bestValAcc,
            'finalMetrics': finalMetrics,
            'trainingHistory': history,
            'checkpointPath': str(self.outputDir / f"best_{modelType}_model.pth")
        }
    
    def crossValidate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        modelType: str,
        modelConfig: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform K-Fold cross-validation.
        
        Args:
            features: Training features
            labels: Training labels
            modelType: Type of model
            modelConfig: Model configuration
            
        Returns:
            Cross-validation results
        """
        self.logger.info(f"Starting {self.numFolds}-Fold Cross-Validation for {modelType}")
        
        # Create dataset
        dataset = NetworkFlowDataset(features, labels)
        
        # Setup K-Fold
        kfold = StratifiedKFold(n_splits=self.numFolds, shuffle=True, random_state=42)
        
        foldResults = []
        
        for fold, (trainIdx, valIdx) in enumerate(kfold.split(features, labels)):
            self.logger.info(f"\nFold {fold + 1}/{self.numFolds}")
            
            # Create data loaders
            trainLoader = DataLoader(
                dataset,
                batch_size=self.batchSize,
                sampler=SubsetRandomSampler(trainIdx)
            )
            valLoader = DataLoader(
                dataset,
                batch_size=self.batchSize,
                sampler=SubsetRandomSampler(valIdx)
            )
            
            # Create and train model
            model = self._createModel(modelType, modelConfig)
            model = model.to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learningRate)
            
            # Train for reduced epochs in cross-validation
            numEpochs = self.numEpochs // 2
            bestAcc = 0.0
            
            for epoch in range(numEpochs):
                trainLoss, trainAcc = self._trainEpoch(model, trainLoader, criterion, optimizer)
                valLoss, valAcc = self._validateEpoch(model, valLoader, criterion)
                
                if valAcc > bestAcc:
                    bestAcc = valAcc
            
            # Final evaluation on validation fold
            metrics = self._evaluateModel(model, valLoader)
            
            foldResults.append({
                'fold': fold + 1,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1Score': metrics['f1Score']
            })
            
            self.logger.info(
                f"Fold {fold + 1} - Accuracy: {metrics['accuracy']:.4f}, "
                f"F1: {metrics['f1Score']:.4f}"
            )
        
        # Calculate average metrics
        avgMetrics = {
            'accuracy': np.mean([r['accuracy'] for r in foldResults]),
            'precision': np.mean([r['precision'] for r in foldResults]),
            'recall': np.mean([r['recall'] for r in foldResults]),
            'f1Score': np.mean([r['f1Score'] for r in foldResults]),
            'std_accuracy': np.std([r['accuracy'] for r in foldResults]),
            'std_f1Score': np.std([r['f1Score'] for r in foldResults])
        }
        
        self.logger.info(
            f"\nCross-Validation Results:\n"
            f"Accuracy: {avgMetrics['accuracy']:.4f} ± {avgMetrics['std_accuracy']:.4f}\n"
            f"F1-Score: {avgMetrics['f1Score']:.4f} ± {avgMetrics['std_f1Score']:.4f}"
        )
        
        # Save results
        resultsPath = self.outputDir / f"{modelType}_cv_results.json"
        with open(resultsPath, 'w') as f:
            json.dump({
                'modelType': modelType,
                'numFolds': self.numFolds,
                'foldResults': foldResults,
                'averageMetrics': avgMetrics
            }, f, indent=2)
        
        return {
            'modelType': modelType,
            'foldResults': foldResults,
            'averageMetrics': avgMetrics,
            'resultsPath': str(resultsPath)
        }
    
    def tuneHyperparameters(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        modelType: str,
        nTrials: int = 50
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna.
        
        Args:
            features: Training features
            labels: Training labels
            modelType: Type of model
            nTrials: Number of optimization trials
            
        Returns:
            Tuning results
        """
        self.logger.info(f"Starting hyperparameter tuning for {modelType} ({nTrials} trials)")
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            
            # Suggest hyperparameters based on model type
            if modelType == 'lstm':
                config = {
                    'inputSize': features.shape[-1],
                    'hiddenSize': trial.suggest_int('hiddenSize', 64, 256, step=64),
                    'numLayers': trial.suggest_int('numLayers', 1, 3),
                    'numClasses': len(np.unique(labels)),
                    'dropoutRate': trial.suggest_float('dropoutRate', 0.1, 0.5),
                    'useAttention': trial.suggest_categorical('useAttention', [True, False])
                }
            else:  # cnn
                config = {
                    'inputChannels': 1,
                    'sequenceLength': features.shape[-1],
                    'numClasses': len(np.unique(labels)),
                    'convLayers': [
                        trial.suggest_int('conv1', 32, 128, step=32),
                        trial.suggest_int('conv2', 64, 256, step=64),
                        trial.suggest_int('conv3', 128, 512, step=128)
                    ],
                    'kernelSizes': [3, 5, 7],
                    'dropoutRate': trial.suggest_float('dropoutRate', 0.1, 0.5)
                }
            
            # Learning rate
            lr = trial.suggest_float('learningRate', 1e-4, 1e-2, log=True)
            
            # Train model with these hyperparameters
            dataset = NetworkFlowDataset(features, labels)
            
            # Split data
            datasetSize = len(dataset)
            indices = list(range(datasetSize))
            split = int(0.2 * datasetSize)
            np.random.shuffle(indices)
            trainIndices, valIndices = indices[split:], indices[:split]
            
            trainLoader = DataLoader(
                dataset, batch_size=self.batchSize,
                sampler=SubsetRandomSampler(trainIndices)
            )
            valLoader = DataLoader(
                dataset, batch_size=self.batchSize,
                sampler=SubsetRandomSampler(valIndices)
            )
            
            # Create model
            model = self._createModel(modelType, config).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # Train for limited epochs
            maxEpochs = 20
            bestValAcc = 0.0
            
            for epoch in range(maxEpochs):
                _, _ = self._trainEpoch(model, trainLoader, criterion, optimizer)
                _, valAcc = self._validateEpoch(model, valLoader, criterion)
                
                if valAcc > bestValAcc:
                    bestValAcc = valAcc
                
                # Optuna pruning
                trial.report(valAcc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return bestValAcc
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(objective, n_trials=nTrials, show_progress_bar=True)
        
        # Get best parameters
        bestParams = study.best_params
        bestScore = study.best_value
        
        self.logger.info(f"\nBest hyperparameters found:")
        for key, value in bestParams.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info(f"Best validation accuracy: {bestScore:.4f}")
        
        # Add loss function to tuning space
        loss_name = trial.suggest_categorical('loss', ['cross_entropy', 'focal_loss'])
        if loss_name == 'focal_loss':
            gamma = trial.suggest_float('gamma', 0.5, 5.0)
            bestParams['loss'] = {'name': 'focal_loss', 'gamma': gamma}
        else:
            bestParams['loss'] = {'name': 'cross_entropy'}

        # Save tuning results
        resultsPath = self.outputDir / f"{modelType}_tuning_results.json"
        with open(resultsPath, 'w') as f:
            json.dump({
                'modelType': modelType,
                'nTrials': nTrials,
                'bestParams': bestParams,
                'bestScore': bestScore,
                'allTrials': [
                    {
                        'number': t.number,
                        'params': t.params,
                        'value': t.value
                    } for t in study.trials if t.value is not None
                ]
            }, f, indent=2)
        
        return {
            'modelType': modelType,
            'bestParams': bestParams,
            'bestScore': bestScore,
            'resultsPath': str(resultsPath)
        }
    
    def _create_loss_function(self, loss_config: Dict[str, Any], data_loader: DataLoader) -> nn.Module:
        """Creates a loss function based on the provided configuration."""
        loss_name = loss_config.get('name', 'cross_entropy').lower()
        alpha = loss_config.get('alpha')
        class_weights = None

        if alpha == 'auto':
            class_weights = self._calculate_class_weights(data_loader)
        elif isinstance(alpha, list):
            class_weights = torch.tensor(alpha, device=self.device)

        if loss_name == 'focal_loss':
            gamma = loss_config.get('gamma', 2.0)
            self.logger.info(f"Using FocalLoss with gamma={gamma} and alpha={alpha}")
            return FocalLoss(alpha=class_weights, gamma=gamma)
        elif loss_name == 'cross_entropy':
            self.logger.info(f"Using CrossEntropyLoss with alpha={alpha}")
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    def _calculate_class_weights(self, data_loader: DataLoader) -> torch.Tensor:
        """Calculates class weights based on the distribution of classes in the dataset."""
        # This is a simplified calculation. For large datasets, consider a more efficient approach.
        all_labels = []
        for _, labels in data_loader:
            all_labels.extend(labels.numpy())
        
        class_counts = np.bincount(all_labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        
        return class_weights.to(self.device)
    
    def _createModel(self, modelType: str, config: Dict[str, Any]) -> nn.Module:
        """Create model based on type and configuration."""
        if modelType == 'lstm':
            return BiLSTMModel(**config)
        elif modelType == 'cnn':
            return CNNModel(**config)
        elif modelType == 'hybrid_cnn_lstm':
            # The HybridCNNLSTM class is passed during initialization
            # Convert parameter names for HybridCNNLSTM
            hybrid_config = config.copy()
            if 'inputFeatures' in hybrid_config:
                hybrid_config['packet_input_size'] = hybrid_config.pop('inputFeatures')
            return self.HybridCNNLSTM(**hybrid_config)
        elif modelType == 'hybrid_cnn_bilstm_attention':
            # The HybridCnnBiLstmAttention class is passed during initialization
            # Convert parameter names for HybridCnnBiLstmAttention
            hybrid_config = config.copy()
            if 'packet_input_size' in hybrid_config:
                hybrid_config['inputFeatures'] = hybrid_config.pop('packet_input_size')
            return self.HybridCnnBiLstmAttention(**hybrid_config)
        else:
            raise ValueError(f"Unknown model type: {modelType}")
    
    def _trainEpoch(
        self,
        model: nn.Module,
        dataLoader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        print(f"[Epoch {epoch}] Starting _trainEpoch...")
        model.train()
        totalLoss = 0.0
        correct = 0
        total = 0

        for batchIdx, (features, labels) in enumerate(dataLoader):
            print(f"[Epoch {epoch}]   _trainEpoch: Processing batch {batchIdx+1}/{len(dataLoader)}")
            features, labels = features.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"[Epoch {epoch}] Finished _trainEpoch.")
        avgLoss = totalLoss / len(dataLoader)
        accuracy = correct / total

        return avgLoss, accuracy

    def _validateEpoch(
        self,
        model: nn.Module,
        dataLoader: DataLoader,
        criterion: nn.Module,
        epoch: int
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        print(f"[Epoch {epoch}] Starting _validateEpoch...")
        model.eval()
        totalLoss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batchIdx, (features, labels) in enumerate(dataLoader):
                print(f"[Epoch {epoch}]   _validateEpoch: Processing batch {batchIdx+1}/{len(dataLoader)}")
                features, labels = features.to(self.device), labels.to(self.device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                totalLoss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"[Epoch {epoch}] Finished _validateEpoch.")
        avgLoss = totalLoss / len(dataLoader)
        accuracy = correct / total

        return avgLoss, accuracy
    
    def _evaluateModel(self, model: nn.Module, dataLoader: DataLoader) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        model.eval()
        allPreds = []
        allLabels = []
        
        with torch.no_grad():
            for features, labels in dataLoader:
                features = features.to(self.device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                
                allPreds.extend(predicted.cpu().numpy())
                allLabels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(allLabels, allPreds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            allLabels, allPreds, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1Score': float(f1)
        }
    
    def _saveCheckpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        score: float,
        path: Path
    ) -> None:
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score
        }, path)
    
    def _loadCheckpoint(self, model: nn.Module, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    def _saveTrainingHistory(self, history: Dict, modelType: str) -> None:
        """Save training history."""
        historyPath = self.outputDir / f"{modelType}_training_history.json"
        with open(historyPath, 'w') as f:
            json.dump(history, f, indent=2)
