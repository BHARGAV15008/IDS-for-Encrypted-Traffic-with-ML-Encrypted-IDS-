"""
Hyperparameter Tuning Module
Implements various hyperparameter optimization techniques:
- Grid Search
- Random Search
- Bayesian Optimization
- Optuna-based optimization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
from itertools import product
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class GridSearchTuner:
    """Grid search for hyperparameter tuning"""
    
    def __init__(
        self,
        modelClass: type,
        paramGrid: Dict[str, List[Any]],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            modelClass: Model class to instantiate
            paramGrid: Dictionary of parameter names and values to try
            device: Device to use
        """
        self.modelClass = modelClass
        self.paramGrid = paramGrid
        self.device = device
        self.results = []
        
    def search(
        self,
        trainLoader: DataLoader,
        valLoader: DataLoader,
        numEpochs: int = 20,
        lossFn: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Perform grid search
        
        Args:
            trainLoader: Training data loader
            valLoader: Validation data loader
            numEpochs: Number of epochs per configuration
            lossFn: Loss function
            
        Returns:
            Best parameters and results
        """
        if lossFn is None:
            lossFn = nn.CrossEntropyLoss()
            
        # Generate all parameter combinations
        paramNames = list(self.paramGrid.keys())
        paramValues = list(self.paramGrid.values())
        combinations = list(product(*paramValues))
        
        logger.info(f"Grid search: {len(combinations)} combinations to try")
        
        bestScore = -float('inf')
        bestParams = None
        
        for idx, combo in enumerate(combinations):
            params = dict(zip(paramNames, combo))
            logger.info(f"Testing combination {idx+1}/{len(combinations)}: {params}")
            
            # Train model with these parameters
            score = self._trainAndEvaluate(
                params, trainLoader, valLoader, numEpochs, lossFn
            )
            
            # Store results
            self.results.append({
                'params': params,
                'score': score
            })
            
            # Update best
            if score > bestScore:
                bestScore = score
                bestParams = params
                
        logger.info(f"Best parameters: {bestParams}, Best score: {bestScore:.4f}")
        
        return {
            'best_params': bestParams,
            'best_score': bestScore,
            'all_results': self.results
        }
        
    def _trainAndEvaluate(
        self,
        params: Dict[str, Any],
        trainLoader: DataLoader,
        valLoader: DataLoader,
        numEpochs: int,
        lossFn: nn.Module
    ) -> float:
        """Train model and return validation score"""
        from .modelTrainer import ModelTrainer
        
        # Extract model parameters
        modelParams = {k: v for k, v in params.items() if k not in ['learningRate', 'batchSize', 'optimizer']}
        
        # Initialize model
        model = self.modelClass(**modelParams)
        trainer = ModelTrainer(model, device=self.device)
        
        # Train
        history = trainer.train(
            trainLoader=trainLoader,
            valLoader=valLoader,
            numEpochs=numEpochs,
            learningRate=params.get('learningRate', 0.001),
            optimizer=params.get('optimizer', 'adam'),
            lossFn=lossFn,
            useEarlyStopping=True,
            patience=5
        )
        
        # Return best validation accuracy
        return max(history['val_acc'])


class RandomSearchTuner:
    """Random search for hyperparameter tuning"""
    
    def __init__(
        self,
        modelClass: type,
        paramDistributions: Dict[str, Callable],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            modelClass: Model class
            paramDistributions: Dictionary of parameter names and sampling functions
            device: Device to use
        """
        self.modelClass = modelClass
        self.paramDistributions = paramDistributions
        self.device = device
        self.results = []
        
    def search(
        self,
        trainLoader: DataLoader,
        valLoader: DataLoader,
        nIterations: int = 50,
        numEpochs: int = 20,
        lossFn: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Perform random search
        
        Args:
            trainLoader: Training data loader
            valLoader: Validation data loader
            nIterations: Number of random configurations to try
            numEpochs: Number of epochs per configuration
            lossFn: Loss function
            
        Returns:
            Best parameters and results
        """
        if lossFn is None:
            lossFn = nn.CrossEntropyLoss()
            
        logger.info(f"Random search: {nIterations} iterations")
        
        bestScore = -float('inf')
        bestParams = None
        
        for iteration in range(nIterations):
            # Sample random parameters
            params = {
                name: sampler() for name, sampler in self.paramDistributions.items()
            }
            
            logger.info(f"Iteration {iteration+1}/{nIterations}: {params}")
            
            # Train and evaluate
            score = self._trainAndEvaluate(
                params, trainLoader, valLoader, numEpochs, lossFn
            )
            
            # Store results
            self.results.append({
                'params': params,
                'score': score
            })
            
            # Update best
            if score > bestScore:
                bestScore = score
                bestParams = params
                
        logger.info(f"Best parameters: {bestParams}, Best score: {bestScore:.4f}")
        
        return {
            'best_params': bestParams,
            'best_score': bestScore,
            'all_results': self.results
        }
        
    def _trainAndEvaluate(
        self,
        params: Dict[str, Any],
        trainLoader: DataLoader,
        valLoader: DataLoader,
        numEpochs: int,
        lossFn: nn.Module
    ) -> float:
        """Train model and return validation score"""
        from .modelTrainer import ModelTrainer
        
        # Extract model parameters
        modelParams = {k: v for k, v in params.items() if k not in ['learningRate', 'batchSize', 'optimizer']}
        
        # Initialize model
        model = self.modelClass(**modelParams)
        trainer = ModelTrainer(model, device=self.device)
        
        # Train
        history = trainer.train(
            trainLoader=trainLoader,
            valLoader=valLoader,
            numEpochs=numEpochs,
            learningRate=params.get('learningRate', 0.001),
            optimizer=params.get('optimizer', 'adam'),
            lossFn=lossFn,
            useEarlyStopping=True,
            patience=5
        )
        
        # Return best validation accuracy
        return max(history['val_acc'])


class BayesianOptimizationTuner:
    """Bayesian optimization for hyperparameter tuning"""
    
    def __init__(
        self,
        modelClass: type,
        paramBounds: Dict[str, Tuple[float, float]],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            modelClass: Model class
            paramBounds: Dictionary of parameter names and (min, max) bounds
            device: Device to use
        """
        self.modelClass = modelClass
        self.paramBounds = paramBounds
        self.device = device
        
    def search(
        self,
        trainLoader: DataLoader,
        valLoader: DataLoader,
        nIterations: int = 30,
        numEpochs: int = 20,
        lossFn: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Perform Bayesian optimization
        
        Args:
            trainLoader: Training data loader
            valLoader: Validation data loader
            nIterations: Number of iterations
            numEpochs: Number of epochs per configuration
            lossFn: Loss function
            
        Returns:
            Best parameters and results
        """
        try:
            from bayes_opt import BayesianOptimization
        except ImportError:
            logger.error("bayesian-optimization not installed. Install with: pip install bayesian-optimization")
            raise
            
        if lossFn is None:
            lossFn = nn.CrossEntropyLoss()
            
        # Store data loaders for objective function
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.numEpochs = numEpochs
        self.lossFn = lossFn
        
        # Create optimizer
        optimizer = BayesianOptimization(
            f=self._objective,
            pbounds=self.paramBounds,
            random_state=42,
            verbose=2
        )
        
        # Run optimization
        optimizer.maximize(
            init_points=5,
            n_iter=nIterations
        )
        
        bestParams = optimizer.max['params']
        bestScore = optimizer.max['target']
        
        logger.info(f"Best parameters: {bestParams}, Best score: {bestScore:.4f}")
        
        return {
            'best_params': bestParams,
            'best_score': bestScore,
            'optimizer': optimizer
        }
        
    def _objective(self, **params) -> float:
        """Objective function for Bayesian optimization"""
        from .modelTrainer import ModelTrainer
        
        # Convert continuous parameters to appropriate types
        processedParams = {}
        for key, value in params.items():
            if 'hidden' in key or 'num' in key or 'layers' in key:
                processedParams[key] = int(value)
            else:
                processedParams[key] = value
                
        # Extract model parameters
        modelParams = {k: v for k, v in processedParams.items() if k not in ['learningRate', 'dropout']}
        
        # Initialize model
        model = self.modelClass(**modelParams)
        trainer = ModelTrainer(model, device=self.device)
        
        # Train
        history = trainer.train(
            trainLoader=self.trainLoader,
            valLoader=self.valLoader,
            numEpochs=self.numEpochs,
            learningRate=processedParams.get('learningRate', 0.001),
            dropout=processedParams.get('dropout', 0.3),
            lossFn=self.lossFn,
            useEarlyStopping=True,
            patience=5
        )
        
        # Return best validation accuracy
        return max(history['val_acc'])


class OptunaTuner:
    """Optuna-based hyperparameter tuning"""
    
    def __init__(
        self,
        modelClass: type,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            modelClass: Model class
            device: Device to use
        """
        self.modelClass = modelClass
        self.device = device
        
    def search(
        self,
        trainLoader: DataLoader,
        valLoader: DataLoader,
        nTrials: int = 50,
        numEpochs: int = 20,
        lossFn: Optional[nn.Module] = None,
        studyName: str = 'ids_optimization'
    ) -> Dict[str, Any]:
        """
        Perform Optuna optimization
        
        Args:
            trainLoader: Training data loader
            valLoader: Validation data loader
            nTrials: Number of trials
            numEpochs: Number of epochs per trial
            lossFn: Loss function
            studyName: Name of the study
            
        Returns:
            Best parameters and results
        """
        try:
            import optuna
        except ImportError:
            logger.error("Optuna not installed. Install with: pip install optuna")
            raise
            
        if lossFn is None:
            lossFn = nn.CrossEntropyLoss()
            
        # Store data for objective function
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.numEpochs = numEpochs
        self.lossFn = lossFn
        
        # Create study
        study = optuna.create_study(
            study_name=studyName,
            direction='maximize',
            sampler=optuna.samplers.TPESampler()
        )
        
        # Run optimization
        study.optimize(self._objective, n_trials=nTrials)
        
        bestParams = study.best_params
        bestScore = study.best_value
        
        logger.info(f"Best parameters: {bestParams}, Best score: {bestScore:.4f}")
        
        return {
            'best_params': bestParams,
            'best_score': bestScore,
            'study': study
        }
        
    def _objective(self, trial) -> float:
        """Objective function for Optuna"""
        from .modelTrainer import ModelTrainer
        
        # Suggest hyperparameters
        params = {
            'inputFeatures': trial.suggest_int('inputFeatures', 32, 128),
            'cnnChannels': [
                trial.suggest_int('cnn_ch1', 32, 128),
                trial.suggest_int('cnn_ch2', 64, 256),
                trial.suggest_int('cnn_ch3', 128, 512)
            ],
            'lstmHiddenSize': trial.suggest_int('lstmHiddenSize', 64, 256),
            'lstmNumLayers': trial.suggest_int('lstmNumLayers', 1, 4),
            'attentionHeads': trial.suggest_categorical('attentionHeads', [4, 8, 16]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learningRate': trial.suggest_float('learningRate', 1e-5, 1e-2, log=True),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        }
        
        # Extract model parameters
        modelParams = {k: v for k, v in params.items() if k not in ['learningRate', 'optimizer']}
        
        # Initialize model
        model = self.modelClass(**modelParams)
        trainer = ModelTrainer(model, device=self.device)
        
        # Train
        history = trainer.train(
            trainLoader=self.trainLoader,
            valLoader=self.valLoader,
            numEpochs=self.numEpochs,
            learningRate=params['learningRate'],
            optimizer=params['optimizer'],
            lossFn=self.lossFn,
            useEarlyStopping=True,
            patience=5
        )
        
        # Return best validation accuracy
        return max(history['val_acc'])


class HyperparameterTunerFactory:
    """Factory for creating hyperparameter tuners"""
    
    @staticmethod
    def create(
        method: str,
        modelClass: type,
        **kwargs
    ):
        """
        Create hyperparameter tuner
        
        Args:
            method: Tuning method ('grid', 'random', 'bayesian', 'optuna')
            modelClass: Model class
            **kwargs: Additional parameters
            
        Returns:
            Hyperparameter tuner
        """
        if method == 'grid':
            return GridSearchTuner(modelClass, **kwargs)
        elif method == 'random':
            return RandomSearchTuner(modelClass, **kwargs)
        elif method == 'bayesian':
            return BayesianOptimizationTuner(modelClass, **kwargs)
        elif method == 'optuna':
            return OptunaTuner(modelClass, **kwargs)
        else:
            raise ValueError(f"Unknown tuning method: {method}")


class AutoMLTuner:
    """
    Automated Machine Learning tuner
    Combines multiple optimization strategies
    """
    
    def __init__(
        self,
        modelClass: type,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            modelClass: Model class
            device: Device to use
        """
        self.modelClass = modelClass
        self.device = device
        
    def autoTune(
        self,
        trainLoader: DataLoader,
        valLoader: DataLoader,
        budget: int = 100,
        strategy: str = 'progressive'
    ) -> Dict[str, Any]:
        """
        Automatically tune hyperparameters with given budget
        
        Args:
            trainLoader: Training data loader
            valLoader: Validation data loader
            budget: Total number of trials
            strategy: Tuning strategy ('progressive', 'aggressive', 'conservative')
            
        Returns:
            Best configuration
        """
        logger.info(f"AutoML tuning with budget={budget}, strategy={strategy}")
        
        if strategy == 'progressive':
            # Start with random search, then Bayesian optimization
            randomBudget = budget // 3
            bayesianBudget = budget - randomBudget
            
            # Random search phase
            logger.info("Phase 1: Random search")
            randomTuner = RandomSearchTuner(
                self.modelClass,
                self._getParamDistributions(),
                self.device
            )
            randomResults = randomTuner.search(
                trainLoader, valLoader, nIterations=randomBudget
            )
            
            # Bayesian optimization phase
            logger.info("Phase 2: Bayesian optimization")
            bayesianTuner = BayesianOptimizationTuner(
                self.modelClass,
                self._getParamBounds(),
                self.device
            )
            bayesianResults = bayesianTuner.search(
                trainLoader, valLoader, nIterations=bayesianBudget
            )
            
            # Return best overall
            if randomResults['best_score'] > bayesianResults['best_score']:
                return randomResults
            else:
                return bayesianResults
                
        elif strategy == 'aggressive':
            # Use Optuna with aggressive pruning
            optunaTuner = OptunaTuner(self.modelClass, self.device)
            return optunaTuner.search(trainLoader, valLoader, nTrials=budget)
            
        elif strategy == 'conservative':
            # Use grid search with limited parameter space
            gridTuner = GridSearchTuner(
                self.modelClass,
                self._getConservativeGrid(),
                self.device
            )
            return gridTuner.search(trainLoader, valLoader)
            
    def _getParamDistributions(self) -> Dict[str, Callable]:
        """Get parameter distributions for random search"""
        return {
            'learningRate': lambda: np.random.uniform(1e-5, 1e-2),
            'dropout': lambda: np.random.uniform(0.1, 0.5),
            'lstmHiddenSize': lambda: np.random.choice([64, 128, 256]),
            'lstmNumLayers': lambda: np.random.choice([1, 2, 3, 4]),
            'attentionHeads': lambda: np.random.choice([4, 8, 16])
        }
        
    def _getParamBounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for Bayesian optimization"""
        return {
            'learningRate': (1e-5, 1e-2),
            'dropout': (0.1, 0.5),
            'lstmHiddenSize': (64, 256),
            'lstmNumLayers': (1, 4)
        }
        
    def _getConservativeGrid(self) -> Dict[str, List[Any]]:
        """Get conservative parameter grid"""
        return {
            'learningRate': [0.001, 0.0001],
            'dropout': [0.3, 0.5],
            'lstmHiddenSize': [128, 256],
            'lstmNumLayers': [2, 3]
        }
