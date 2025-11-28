"""
Ensemble Module for Robust Classification
Combines deep learning features with traditional ML classifiers
Implements voting, stacking, and boosting ensemble methods
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Tuple
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import logging

logger = logging.getLogger(__name__)


class EnsembleClassifier(nn.Module):
    """
    Ensemble classifier combining deep learning features with ML models
    """
    
    def __init__(
        self,
        featureDim: int,
        numClasses: int,
        ensembleType: str = 'voting',
        baseModels: Optional[List[str]] = None
    ):
        """
        Args:
            featureDim: Dimension of input features
            numClasses: Number of output classes
            ensembleType: Type of ensemble ('voting', 'stacking', 'weighted')
            baseModels: List of base model names
        """
        super().__init__()
        
        self.featureDim = featureDim
        self.numClasses = numClasses
        self.ensembleType = ensembleType
        
        if baseModels is None:
            baseModels = ['random_forest', 'gradient_boosting', 'svm']
        
        self.baseModels = baseModels
        self.models = {}
        self.isFitted = False
        
        # Initialize base models
        self._initializeModels()
        
        logger.info(f"Ensemble Classifier initialized: type={ensembleType}, models={baseModels}")
        
    def _initializeModels(self):
        """Initialize base ML models"""
        for modelName in self.baseModels:
            if modelName == 'random_forest':
                self.models[modelName] = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
            elif modelName == 'gradient_boosting':
                self.models[modelName] = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42
                )
            elif modelName == 'svm':
                self.models[modelName] = SVC(
                    kernel='rbf',
                    C=1.0,
                    probability=True,
                    random_state=42
                )
            elif modelName == 'logistic':
                self.models[modelName] = LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    n_jobs=-1
                )
            elif modelName == 'decision_tree':
                self.models[modelName] = DecisionTreeClassifier(
                    max_depth=15,
                    random_state=42
                )
                
    def fit(self, features: np.ndarray, labels: np.ndarray):
        """
        Fit ensemble models
        
        Args:
            features: Training features (n_samples, feature_dim)
            labels: Training labels (n_samples,)
        """
        logger.info(f"Fitting ensemble models on {len(features)} samples")
        
        if self.ensembleType == 'voting':
            # Fit individual models
            for name, model in self.models.items():
                logger.info(f"Fitting {name}...")
                model.fit(features, labels)
                
        elif self.ensembleType == 'stacking':
            # Create stacking ensemble
            estimators = [(name, model) for name, model in self.models.items()]
            self.stackingModel = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5,
                n_jobs=-1
            )
            self.stackingModel.fit(features, labels)
            
        self.isFitted = True
        logger.info("Ensemble fitting complete")
        
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict classes
        
        Args:
            features: Input features (n_samples, feature_dim)
            
        Returns:
            predictions: Predicted classes (n_samples,)
        """
        if not self.isFitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
            
        if self.ensembleType == 'voting':
            # Majority voting
            predictions = []
            for model in self.models.values():
                predictions.append(model.predict(features))
            predictions = np.array(predictions)
            
            # Majority vote
            finalPredictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=predictions
            )
            return finalPredictions
            
        elif self.ensembleType == 'stacking':
            return self.stackingModel.predict(features)
            
    def predictProba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            features: Input features (n_samples, feature_dim)
            
        Returns:
            probabilities: Class probabilities (n_samples, num_classes)
        """
        if not self.isFitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
            
        if self.ensembleType == 'voting':
            # Average probabilities
            probas = []
            for model in self.models.values():
                probas.append(model.predict_proba(features))
            return np.mean(probas, axis=0)
            
        elif self.ensembleType == 'stacking':
            return self.stackingModel.predict_proba(features)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (for PyTorch compatibility)
        
        Args:
            x: Input tensor (batch, feature_dim)
            
        Returns:
            predictions: (batch, num_classes)
        """
        # Convert to numpy
        features = x.detach().cpu().numpy()
        
        # Predict probabilities
        probas = self.predictProba(features)
        
        # Convert back to tensor
        return torch.from_numpy(probas).to(x.device)


class DeepEnsemble(nn.Module):
    """
    Deep ensemble combining multiple neural network models
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        ensembleMethod: str = 'average'
    ):
        """
        Args:
            models: List of PyTorch models
            ensembleMethod: Ensemble method ('average', 'weighted', 'voting')
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.ensembleMethod = ensembleMethod
        self.numModels = len(models)
        
        # Learnable weights for weighted ensemble
        if ensembleMethod == 'weighted':
            self.weights = nn.Parameter(torch.ones(self.numModels) / self.numModels)
            
        logger.info(f"Deep Ensemble initialized: {self.numModels} models, method={ensembleMethod}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble prediction
        """
        # Get predictions from all models
        predictions = [model(x) for model in self.models]
        predictions = torch.stack(predictions)  # (num_models, batch, num_classes)
        
        if self.ensembleMethod == 'average':
            # Simple average
            output = predictions.mean(dim=0)
            
        elif self.ensembleMethod == 'weighted':
            # Weighted average
            weights = torch.softmax(self.weights, dim=0)
            output = (predictions * weights.view(-1, 1, 1)).sum(dim=0)
            
        elif self.ensembleMethod == 'voting':
            # Hard voting
            predictions = torch.argmax(predictions, dim=-1)  # (num_models, batch)
            output = torch.mode(predictions, dim=0).values
            
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensembleMethod}")
            
        return output


class HybridDeepEnsemble(nn.Module):
    """
    Hybrid ensemble combining deep learning model with traditional ML
    """
    
    def __init__(
        self,
        deepModel: nn.Module,
        featureDim: int,
        numClasses: int,
        mlModels: Optional[List[str]] = None
    ):
        """
        Args:
            deepModel: Deep learning model for feature extraction
            featureDim: Feature dimension from deep model
            numClasses: Number of classes
            mlModels: List of ML model names
        """
        super().__init__()
        
        self.deepModel = deepModel
        self.featureDim = featureDim
        self.numClasses = numClasses
        
        # ML ensemble
        self.mlEnsemble = EnsembleClassifier(
            featureDim=featureDim,
            numClasses=numClasses,
            ensembleType='stacking',
            baseModels=mlModels
        )
        
        logger.info("Hybrid Deep Ensemble initialized")
        
    def extractFeatures(self, x: torch.Tensor) -> np.ndarray:
        """Extract features using deep model"""
        self.deepModel.eval()
        with torch.no_grad():
            if hasattr(self.deepModel, 'extractFeatures'):
                features = self.deepModel.extractFeatures(x)
            else:
                features = self.deepModel(x)
        return features.cpu().numpy()
        
    def fit(self, x: torch.Tensor, labels: np.ndarray):
        """
        Fit ML ensemble on deep features
        
        Args:
            x: Input tensor
            labels: Training labels
        """
        features = self.extractFeatures(x)
        self.mlEnsemble.fit(features, labels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.extractFeatures(x)
        probas = self.mlEnsemble.predictProba(features)
        return torch.from_numpy(probas).to(x.device)
        
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Predict classes"""
        features = self.extractFeatures(x)
        return self.mlEnsemble.predict(features)


class AdaptiveEnsemble(nn.Module):
    """
    Adaptive ensemble that learns to weight different models
    based on input characteristics
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        inputDim: int,
        numClasses: int
    ):
        """
        Args:
            models: List of base models
            inputDim: Input dimension
            numClasses: Number of classes
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.numModels = len(models)
        self.numClasses = numClasses
        
        # Gating network to compute adaptive weights
        self.gatingNetwork = nn.Sequential(
            nn.Linear(inputDim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.numModels),
            nn.Softmax(dim=-1)
        )
        
        logger.info(f"Adaptive Ensemble initialized: {self.numModels} models")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive weighting
        
        Args:
            x: Input tensor (batch, seq_len, features) or (batch, features)
            
        Returns:
            Weighted ensemble prediction
        """
        # Compute gating weights
        if x.dim() == 3:
            gatingInput = x.mean(dim=1)  # Pool sequence dimension
        else:
            gatingInput = x
            
        weights = self.gatingNetwork(gatingInput)  # (batch, num_models)
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)  # (batch, num_classes)
            predictions.append(pred)
        predictions = torch.stack(predictions, dim=1)  # (batch, num_models, num_classes)
        
        # Apply adaptive weights
        weights = weights.unsqueeze(-1)  # (batch, num_models, 1)
        output = (predictions * weights).sum(dim=1)  # (batch, num_classes)
        
        return output


class BoostingEnsemble:
    """
    Boosting ensemble for sequential model training
    """
    
    def __init__(
        self,
        baseModelClass: type,
        numModels: int = 5,
        learningRate: float = 0.1
    ):
        """
        Args:
            baseModelClass: Class of base model
            numModels: Number of boosting iterations
            learningRate: Learning rate for boosting
        """
        self.baseModelClass = baseModelClass
        self.numModels = numModels
        self.learningRate = learningRate
        self.models = []
        self.modelWeights = []
        
    def fit(self, xTrain: torch.Tensor, yTrain: torch.Tensor):
        """
        Fit boosting ensemble
        
        Args:
            xTrain: Training features
            yTrain: Training labels
        """
        # Initialize sample weights
        sampleWeights = torch.ones(len(yTrain)) / len(yTrain)
        
        for i in range(self.numModels):
            logger.info(f"Training boosting model {i+1}/{self.numModels}")
            
            # Train base model
            model = self.baseModelClass()
            # Training logic here (simplified)
            
            # Compute model weight based on error
            # This is a simplified version
            modelWeight = 1.0
            
            self.models.append(model)
            self.modelWeights.append(modelWeight)
            
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using boosting ensemble"""
        predictions = []
        for model, weight in zip(self.models, self.modelWeights):
            pred = model(x)
            predictions.append(pred * weight)
        return torch.stack(predictions).sum(dim=0)


class EnsembleModelFactory:
    """Factory for creating ensemble models"""
    
    @staticmethod
    def create(
        ensembleType: str,
        **kwargs
    ) -> nn.Module:
        """
        Create ensemble model
        
        Args:
            ensembleType: Type of ensemble ('ml', 'deep', 'hybrid', 'adaptive')
            **kwargs: Additional parameters
            
        Returns:
            Ensemble model
        """
        if ensembleType == 'ml':
            return EnsembleClassifier(**kwargs)
        elif ensembleType == 'deep':
            return DeepEnsemble(**kwargs)
        elif ensembleType == 'hybrid':
            return HybridDeepEnsemble(**kwargs)
        elif ensembleType == 'adaptive':
            return AdaptiveEnsemble(**kwargs)
        else:
            raise ValueError(f"Unknown ensemble type: {ensembleType}")
