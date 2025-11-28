"""
Model Training Module
Comprehensive training framework with hyperparameter tuning, early stopping,
learning rate scheduling, and advanced optimization techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List, Tuple, Callable, Union
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

from customLosses import FocalLoss

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(
        self,
        patience: int = 10,
        minDelta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            minDelta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for loss or accuracy
        """
        self.patience = patience
        self.minDelta = minDelta
        self.mode = mode
        self.counter = 0
        self.bestScore = None
        self.shouldStop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current metric value
            
        Returns:
            True if should stop
        """
        if self.bestScore is None:
            self.bestScore = score
            return False
            
        if self.mode == 'min':
            improved = score < (self.bestScore - self.minDelta)
        else:
            improved = score > (self.bestScore + self.minDelta)
            
        if improved:
            self.bestScore = score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.shouldStop = True
            logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            
        return self.shouldStop


class LearningRateScheduler:
    """Custom learning rate scheduler"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        mode: str = 'plateau',
        factor: float = 0.5,
        patience: int = 5,
        minLr: float = 1e-6
    ):
        """
        Args:
            optimizer: Optimizer
            mode: Scheduler mode ('plateau', 'step', 'cosine')
            factor: Factor to reduce LR
            patience: Patience for plateau mode
            minLr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.minLr = minLr
        self.counter = 0
        self.bestScore = None
        
        if mode == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=factor)
        elif mode == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        elif mode == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience, min_lr=minLr
            )
            
    def step(self, metric: Optional[float] = None):
        """Update learning rate"""
        if self.mode == 'plateau' and metric is not None:
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
            
    def getCurrentLr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


class ModelTrainer:
    """
    Comprehensive model trainer with advanced features
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpointDir: str = '03_Models/03_Trained_Weights',
        logDir: str = 'logs'
    ):
        """
        Args:
            model: Model to train
            device: Device to use
            checkpointDir: Directory to save checkpoints
            logDir: Directory for logs
        """
        self.model = model.to(device)
        self.device = device
        self.checkpointDir = Path(checkpointDir)
        self.logDir = Path(logDir)
        
        self.checkpointDir.mkdir(parents=True, exist_ok=True)
        self.logDir.mkdir(parents=True, exist_ok=True)
        
        self.trainingHistory = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        logger.info(f"Model Trainer initialized on device: {self.device}")
        
    def _calculate_class_weights(self, data_loader: DataLoader) -> Optional[torch.Tensor]:
        """Calculate class weights based on inverse frequency."""
        all_targets = []
        for _, targets in data_loader:
            all_targets.append(targets)
        all_targets = torch.cat(all_targets)
        
        class_counts = torch.bincount(all_targets)
        if len(class_counts) == 0:
            logger.warning("Could not calculate class weights, no targets found.")
            return None
            
        total_samples = all_targets.size(0)
        
        weights = total_samples / (len(class_counts) * class_counts.float())
        weights = weights / weights.sum() # Normalize
        
        logger.info(f"Calculated class weights: {weights.tolist()}")
        return weights.to(self.device)

    def train(
        self,
        trainLoader: DataLoader,
        valLoader: Optional[DataLoader] = None,
        numEpochs: int = 100,
        learningRate: float = 0.001,
        optimizer: str = 'adamw',
        loss_config: Optional[Dict] = None,
        useEarlyStopping: bool = True,
        patience: int = 10,
        useLrScheduler: bool = True,
        schedulerMode: str = 'plateau',
        gradientClipping: Optional[float] = 1.0,
        mixedPrecision: bool = False,
        saveEvery: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            trainLoader: Training data loader
            valLoader: Validation data loader
            numEpochs: Number of epochs
            learningRate: Initial learning rate
            optimizer: Optimizer type ('adam', 'sgd', 'adamw')
            loss_config: Configuration for the loss function.
                         Example: {'name': 'focal_loss', 'gamma': 2.0, 'alpha': 'auto'}
                                  {'name': 'cross_entropy', 'alpha': 'auto'}
            useEarlyStopping: Whether to use early stopping
            patience: Early stopping patience
            useLrScheduler: Whether to use LR scheduler
            schedulerMode: LR scheduler mode
            gradientClipping: Gradient clipping value
            mixedPrecision: Use mixed precision training
            saveEvery: Save checkpoint every N epochs
            
        Returns:
            Training history
        """
        # Initialize optimizer
        if optimizer == 'adam':
            opt = optim.Adam(self.model.parameters(), lr=learningRate)
        elif optimizer == 'sgd':
            opt = optim.SGD(self.model.parameters(), lr=learningRate, momentum=0.9)
        elif optimizer == 'adamw':
            opt = optim.AdamW(self.model.parameters(), lr=learningRate, weight_decay=0.01)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Initialize loss function
        if loss_config is None:
            loss_config = {'name': 'cross_entropy'}
            
        loss_name = loss_config.get('name', 'cross_entropy').lower()
        alpha = loss_config.get('alpha')
        class_weights = None

        if alpha == 'auto':
            class_weights = self._calculate_class_weights(trainLoader)
        elif isinstance(alpha, list):
            class_weights = torch.tensor(alpha, device=self.device)

        if loss_name == 'focal_loss':
            gamma = loss_config.get('gamma', 2.0)
            lossFn = FocalLoss(alpha=class_weights, gamma=gamma)
            logger.info(f"Using FocalLoss with gamma={gamma} and alpha={alpha}")
        elif loss_name == 'cross_entropy':
            lossFn = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Using CrossEntropyLoss with alpha={alpha}")
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
            
        # Initialize early stopping
        earlyStopping = None
        if useEarlyStopping:
            earlyStopping = EarlyStopping(patience=patience, mode='min')
            
        # Initialize LR scheduler
        lrScheduler = None
        if useLrScheduler:
            lrScheduler = LearningRateScheduler(opt, mode=schedulerMode)
            
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if mixedPrecision and self.device == 'cuda' else None
        
        # Training loop
        bestValLoss = float('inf')
        
        for epoch in range(numEpochs):
            # Training phase
            trainLoss, trainAcc = self._trainEpoch(
                trainLoader, opt, lossFn, gradientClipping, scaler
            )
            
            # Validation phase
            if valLoader is not None:
                valLoss, valAcc = self._validateEpoch(valLoader, lossFn)
            else:
                valLoss, valAcc = trainLoss, trainAcc
                
            # Update history
            self.trainingHistory['train_loss'].append(trainLoss)
            self.trainingHistory['val_loss'].append(valLoss)
            self.trainingHistory['train_acc'].append(trainAcc)
            self.trainingHistory['val_acc'].append(valAcc)
            self.trainingHistory['learning_rate'].append(lrScheduler.getCurrentLr() if lrScheduler else learningRate)
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{numEpochs} - "
                f"Train Loss: {trainLoss:.4f}, Train Acc: {trainAcc:.4f}, "
                f"Val Loss: {valLoss:.4f}, Val Acc: {valAcc:.4f}, "
                f"LR: {self.trainingHistory['learning_rate'][-1]:.6f}"
            )
            
            # Update learning rate
            if lrScheduler is not None:
                lrScheduler.step(valLoss)
                
            # Save best model
            if valLoss < bestValLoss:
                bestValLoss = valLoss
                self.saveCheckpoint('best_model.pth', epoch, opt, valLoss)
                
            # Save periodic checkpoint
            if (epoch + 1) % saveEvery == 0:
                self.saveCheckpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, opt, valLoss)
                
            # Early stopping
            if earlyStopping is not None and earlyStopping(valLoss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        # Save final model
        self.saveCheckpoint('final_model.pth', numEpochs, opt, valLoss)
        
        # Save training history
        self._saveHistory()
        
        return self.trainingHistory
        
    def _trainEpoch(
        self,
        dataLoader: DataLoader,
        optimizer: optim.Optimizer,
        lossFn: nn.Module,
        gradientClipping: Optional[float],
        scaler: Optional[torch.cuda.amp.GradScaler]
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        totalLoss = 0.0
        correct = 0
        total = 0
        
        for batchIdx, (inputs, targets) in enumerate(dataLoader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = lossFn(outputs, targets)
                    
                scaler.scale(loss).backward()
                
                if gradientClipping is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradientClipping)
                    
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self.model(inputs)
                loss = lossFn(outputs, targets)
                loss.backward()
                
                if gradientClipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradientClipping)
                    
                optimizer.step()
                
            # Statistics
            totalLoss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        avgLoss = totalLoss / len(dataLoader)
        accuracy = correct / total if total > 0 else 0
        
        return avgLoss, accuracy
        
    def _validateEpoch(
        self,
        dataLoader: DataLoader,
        lossFn: nn.Module
    ) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        totalLoss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataLoader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = lossFn(outputs, targets)
                
                totalLoss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        avgLoss = totalLoss / len(dataLoader)
        accuracy = correct / total if total > 0 else 0
        
        return avgLoss, accuracy
        
    def saveCheckpoint(
        self,
        filename: str,
        epoch: int,
        optimizer: optim.Optimizer,
        loss: float
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'training_history': self.trainingHistory
        }
        
        path = self.checkpointDir / filename
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        
    def loadCheckpoint(self, filename: str, optimizer: Optional[optim.Optimizer] = None):
        """Load model checkpoint"""
        path = self.checkpointDir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if 'training_history' in checkpoint:
            self.trainingHistory = checkpoint['training_history']
            
        logger.info(f"Checkpoint loaded: {path}")
        
        return checkpoint.get('epoch', 0), checkpoint.get('loss', 0)
        
    def _saveHistory(self):
        """Save training history to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        historyPath = self.logDir / f'training_history_{timestamp}.json'
        
        with open(historyPath, 'w') as f:
            json.dump(self.trainingHistory, f, indent=4)
            
        logger.info(f"Training history saved: {historyPath}")
        
    def evaluate(
        self,
        testLoader: DataLoader,
        lossFn: Optional[nn.Module] = None
    ) -> Dict[str, Union[float, list]]:
        """
        Evaluate model on test set
        
        Args:
            testLoader: Test data loader
            lossFn: Loss function
            
        Returns:
            Dictionary of metrics
        """
        if lossFn is None:
            lossFn = nn.CrossEntropyLoss()
            
        self.model.eval()
        totalLoss = 0.0
        
        all_preds = []
        all_targets = []
        all_probas = []
        
        with torch.no_grad():
            for inputs, targets in testLoader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = lossFn(outputs, targets)
                
                totalLoss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                probas = torch.nn.functional.softmax(outputs, dim=1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probas.extend(probas.cpu().numpy())
                
        avg_loss = totalLoss / len(testLoader)
        accuracy = accuracy_score(all_targets, all_preds)
        
        metrics = {
            'test_loss': avg_loss,
            'test_acc': accuracy,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probas
        }
        
        logger.info(f"Test Loss: {metrics['test_loss']:.4f}, Test Acc: {metrics['test_acc']:.4f}")
        
        return metrics


class CrossValidator:
    """K-Fold Cross Validation"""
    
    def __init__(
        self,
        modelClass: type,
        modelKwargs: Dict,
        kFolds: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            modelClass: Model class
            modelKwargs: Model initialization arguments
            kFolds: Number of folds
            device: Device to use
        """
        self.modelClass = modelClass
        self.modelKwargs = modelKwargs
        self.kFolds = kFolds
        self.device = device
        
    def run(
        self,
        xData: torch.Tensor,
        yData: torch.Tensor,
        **trainKwargs
    ) -> Dict[str, List[float]]:
        """
        Run k-fold cross validation
        
        Args:
            xData: Input data
            yData: Labels
            **trainKwargs: Training arguments
            
        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import KFold
        
        kfold = KFold(n_splits=self.kFolds, shuffle=True, random_state=42)
        
        foldResults = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for fold, (trainIdx, valIdx) in enumerate(kfold.split(xData)):
            logger.info(f"Training fold {fold+1}/{self.kFolds}")
            
            # Create data loaders
            trainDataset = TensorDataset(xData[trainIdx], yData[trainIdx])
            valDataset = TensorDataset(xData[valIdx], yData[valIdx])
            
            trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)
            valLoader = DataLoader(valDataset, batch_size=64)
            
            # Initialize model
            model = self.modelClass(**self.modelKwargs)
            trainer = ModelTrainer(model, device=self.device)
            
            # Train
            history = trainer.train(trainLoader, valLoader, **trainKwargs)
            
            # Store results
            foldResults['train_loss'].append(history['train_loss'][-1])
            foldResults['val_loss'].append(history['val_loss'][-1])
            foldResults['train_acc'].append(history['train_acc'][-1])
            foldResults['val_acc'].append(history['val_acc'][-1])
            
        # Compute averages
        avgResults = {
            'avg_train_loss': np.mean(foldResults['train_loss']),
            'avg_val_loss': np.mean(foldResults['val_loss']),
            'avg_train_acc': np.mean(foldResults['train_acc']),
            'avg_val_acc': np.mean(foldResults['val_acc']),
            'std_val_acc': np.std(foldResults['val_acc'])
        }
        
        logger.info(f"Cross-validation results: {avgResults}")
        
        return avgResults
