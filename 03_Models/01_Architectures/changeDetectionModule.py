"""
Change Detection Module
Detects small changes and new patterns in network traffic
Implements drift detection, anomaly detection, and pattern discovery
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect concept drift in network traffic patterns"""
    
    def __init__(
        self,
        windowSize: int = 1000,
        driftThreshold: float = 0.1,
        warningThreshold: float = 0.05
    ):
        """
        Args:
            windowSize: Size of sliding window
            driftThreshold: Threshold for drift detection
            warningThreshold: Threshold for warning
        """
        self.windowSize = windowSize
        self.driftThreshold = driftThreshold
        self.warningThreshold = warningThreshold
        
        self.referenceWindow = deque(maxlen=windowSize)
        self.currentWindow = deque(maxlen=windowSize)
        self.driftDetected = False
        self.warningDetected = False
        
    def update(self, prediction: int, actual: int) -> Dict[str, bool]:
        """
        Update drift detector with new prediction
        
        Args:
            prediction: Predicted label
            actual: Actual label
            
        Returns:
            Dictionary with drift and warning flags
        """
        error = 1 if prediction != actual else 0
        
        # Add to current window
        self.currentWindow.append(error)
        
        # Check if we have enough data
        if len(self.currentWindow) < self.windowSize // 2:
            return {'drift': False, 'warning': False}
            
        # Calculate error rates
        if len(self.referenceWindow) > 0:
            refErrorRate = np.mean(self.referenceWindow)
            currErrorRate = np.mean(self.currentWindow)
            
            # Calculate drift
            errorDiff = abs(currErrorRate - refErrorRate)
            
            if errorDiff > self.driftThreshold:
                self.driftDetected = True
                self.warningDetected = False
                # Reset reference window
                self.referenceWindow = deque(self.currentWindow, maxlen=self.windowSize)
                self.currentWindow.clear()
                logger.warning(f"Drift detected! Error difference: {errorDiff:.4f}")
                
            elif errorDiff > self.warningThreshold:
                self.warningDetected = True
                logger.info(f"Drift warning! Error difference: {errorDiff:.4f}")
            else:
                self.warningDetected = False
                
        else:
            # Initialize reference window
            self.referenceWindow = deque(self.currentWindow, maxlen=self.windowSize)
            
        return {
            'drift': self.driftDetected,
            'warning': self.warningDetected
        }
        
    def reset(self):
        """Reset drift detector"""
        self.referenceWindow.clear()
        self.currentWindow.clear()
        self.driftDetected = False
        self.warningDetected = False


class NoveltyDetector(nn.Module):
    """Detect novel patterns using autoencoder-based approach"""
    
    def __init__(
        self,
        inputDim: int,
        hiddenDims: List[int] = [128, 64, 32],
        threshold: Optional[float] = None
    ):
        """
        Args:
            inputDim: Input feature dimension
            hiddenDims: Hidden layer dimensions
            threshold: Reconstruction error threshold (auto-computed if None)
        """
        super().__init__()
        
        self.inputDim = inputDim
        self.threshold = threshold
        self.reconstructionErrors = []
        
        # Encoder
        encoderLayers = []
        prevDim = inputDim
        for hiddenDim in hiddenDims:
            encoderLayers.extend([
                nn.Linear(prevDim, hiddenDim),
                nn.ReLU(),
                nn.BatchNorm1d(hiddenDim),
                nn.Dropout(0.2)
            ])
            prevDim = hiddenDim
            
        self.encoder = nn.Sequential(*encoderLayers)
        
        # Decoder
        decoderLayers = []
        for i in range(len(hiddenDims) - 1, -1, -1):
            outDim = hiddenDims[i - 1] if i > 0 else inputDim
            decoderLayers.extend([
                nn.Linear(prevDim, outDim),
                nn.ReLU() if i > 0 else nn.Identity(),
                nn.BatchNorm1d(outDim) if i > 0 else nn.Identity(),
                nn.Dropout(0.2) if i > 0 else nn.Identity()
            ])
            prevDim = outDim
            
        self.decoder = nn.Sequential(*decoderLayers)
        
        logger.info(f"Novelty Detector initialized: input_dim={inputDim}, hidden_dims={hiddenDims}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
    def computeReconstructionError(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error"""
        reconstructed = self.forward(x)
        error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error
        
    def fitThreshold(self, xTrain: torch.Tensor, percentile: float = 95):
        """
        Fit threshold based on training data
        
        Args:
            xTrain: Training data
            percentile: Percentile for threshold
        """
        self.eval()
        with torch.no_grad():
            errors = self.computeReconstructionError(xTrain)
            self.threshold = np.percentile(errors.cpu().numpy(), percentile)
            
        logger.info(f"Novelty threshold set to: {self.threshold:.6f}")
        
    def detectNovelty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect novel patterns
        
        Args:
            x: Input data
            
        Returns:
            isNovel: Boolean tensor indicating novelty
            errors: Reconstruction errors
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call fitThreshold() first.")
            
        self.eval()
        with torch.no_grad():
            errors = self.computeReconstructionError(x)
            isNovel = errors > self.threshold
            
        return isNovel, errors


class IncrementalLearner(nn.Module):
    """Incremental learning for adapting to new patterns"""
    
    def __init__(
        self,
        baseModel: nn.Module,
        memorySize: int = 1000,
        updateFrequency: int = 100
    ):
        """
        Args:
            baseModel: Base model to adapt
            memorySize: Size of experience replay buffer
            updateFrequency: Frequency of model updates
        """
        super().__init__()
        
        self.baseModel = baseModel
        self.memorySize = memorySize
        self.updateFrequency = updateFrequency
        
        self.memory = {
            'features': deque(maxlen=memorySize),
            'labels': deque(maxlen=memorySize)
        }
        
        self.updateCounter = 0
        
    def addToMemory(self, features: torch.Tensor, labels: torch.Tensor):
        """Add samples to memory"""
        self.memory['features'].append(features.cpu())
        self.memory['labels'].append(labels.cpu())
        
    def incrementalUpdate(
        self,
        optimizer: torch.optim.Optimizer,
        lossFn: nn.Module,
        device: str = 'cuda'
    ):
        """Perform incremental update"""
        if len(self.memory['features']) < 32:
            return
            
        self.baseModel.train()
        
        # Sample from memory
        indices = np.random.choice(len(self.memory['features']), min(32, len(self.memory['features'])), replace=False)
        
        batchFeatures = torch.stack([self.memory['features'][i] for i in indices]).to(device)
        batchLabels = torch.stack([self.memory['labels'][i] for i in indices]).to(device)
        
        # Update model
        optimizer.zero_grad()
        outputs = self.baseModel(batchFeatures)
        loss = lossFn(outputs, batchLabels)
        loss.backward()
        optimizer.step()
        
        logger.debug(f"Incremental update: loss={loss.item():.4f}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.baseModel(x)


class PatternDiscovery:
    """Discover new patterns using clustering and statistical methods"""
    
    def __init__(
        self,
        numClusters: int = 10,
        updateInterval: int = 1000
    ):
        """
        Args:
            numClusters: Number of clusters for pattern discovery
            updateInterval: Interval for updating clusters
        """
        self.numClusters = numClusters
        self.updateInterval = updateInterval
        self.clusterCenters = None
        self.sampleBuffer = []
        self.updateCounter = 0
        
    def addSample(self, features: np.ndarray):
        """Add sample to buffer"""
        self.sampleBuffer.append(features)
        self.updateCounter += 1
        
        if self.updateCounter >= self.updateInterval:
            self.updateClusters()
            self.updateCounter = 0
            
    def updateClusters(self):
        """Update cluster centers"""
        if len(self.sampleBuffer) < self.numClusters:
            return
            
        from sklearn.cluster import KMeans
        
        X = np.array(self.sampleBuffer)
        kmeans = KMeans(n_clusters=self.numClusters, random_state=42)
        kmeans.fit(X)
        
        # Check for new patterns
        if self.clusterCenters is not None:
            # Compute distance between old and new centers
            from scipy.spatial.distance import cdist
            distances = cdist(self.clusterCenters, kmeans.cluster_centers_)
            minDistances = distances.min(axis=0)
            
            # Identify new patterns (clusters far from previous centers)
            newPatterns = minDistances > np.percentile(minDistances, 75)
            
            if newPatterns.any():
                logger.info(f"Discovered {newPatterns.sum()} new patterns")
                
        self.clusterCenters = kmeans.cluster_centers_
        self.sampleBuffer.clear()
        
    def detectNewPattern(self, features: np.ndarray) -> bool:
        """Detect if sample represents a new pattern"""
        if self.clusterCenters is None:
            return False
            
        # Compute distance to nearest cluster
        distances = np.linalg.norm(self.clusterCenters - features, axis=1)
        minDistance = distances.min()
        
        # Compare to typical distances
        threshold = np.mean(distances) + 2 * np.std(distances)
        
        return minDistance > threshold


class AdaptiveThresholdDetector:
    """Adaptive threshold for detecting anomalies with changing baselines"""
    
    def __init__(
        self,
        windowSize: int = 500,
        numStd: float = 3.0,
        adaptationRate: float = 0.1
    ):
        """
        Args:
            windowSize: Size of sliding window
            numStd: Number of standard deviations for threshold
            adaptationRate: Rate of threshold adaptation
        """
        self.windowSize = windowSize
        self.numStd = numStd
        self.adaptationRate = adaptationRate
        
        self.scoreWindow = deque(maxlen=windowSize)
        self.threshold = None
        
    def update(self, score: float) -> bool:
        """
        Update detector and check for anomaly
        
        Args:
            score: Anomaly score
            
        Returns:
            True if anomaly detected
        """
        self.scoreWindow.append(score)
        
        if len(self.scoreWindow) < 50:
            return False
            
        # Compute adaptive threshold
        mean = np.mean(self.scoreWindow)
        std = np.std(self.scoreWindow)
        newThreshold = mean + self.numStd * std
        
        # Adapt threshold gradually
        if self.threshold is None:
            self.threshold = newThreshold
        else:
            self.threshold = (1 - self.adaptationRate) * self.threshold + self.adaptationRate * newThreshold
            
        # Detect anomaly
        isAnomaly = score > self.threshold
        
        return isAnomaly


class EnsembleChangeDetector:
    """Ensemble of change detection methods for robust detection"""
    
    def __init__(
        self,
        inputDim: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            inputDim: Input feature dimension
            device: Device to use
        """
        self.device = device
        
        # Initialize detectors
        self.driftDetector = DriftDetector()
        self.noveltyDetector = NoveltyDetector(inputDim).to(device)
        self.patternDiscovery = PatternDiscovery()
        self.adaptiveThreshold = AdaptiveThresholdDetector()
        
        logger.info("Ensemble Change Detector initialized")
        
    def detect(
        self,
        features: torch.Tensor,
        prediction: Optional[int] = None,
        actual: Optional[int] = None
    ) -> Dict[str, bool]:
        """
        Detect changes using ensemble of methods
        
        Args:
            features: Input features
            prediction: Predicted label (optional)
            actual: Actual label (optional)
            
        Returns:
            Dictionary of detection results
        """
        results = {}
        
        # Drift detection
        if prediction is not None and actual is not None:
            driftResults = self.driftDetector.update(prediction, actual)
            results['drift'] = driftResults['drift']
            results['drift_warning'] = driftResults['warning']
        
        # Novelty detection
        if self.noveltyDetector.threshold is not None:
            isNovel, error = self.noveltyDetector.detectNovelty(features)
            results['novelty'] = isNovel.any().item()
            results['novelty_score'] = error.mean().item()
            
            # Adaptive threshold
            results['adaptive_anomaly'] = self.adaptiveThreshold.update(error.mean().item())
        
        # Pattern discovery
        featuresNp = features.cpu().numpy()
        if featuresNp.ndim == 2:
            featuresNp = featuresNp[0]
        self.patternDiscovery.addSample(featuresNp)
        results['new_pattern'] = self.patternDiscovery.detectNewPattern(featuresNp)
        
        # Ensemble decision (majority voting)
        detectionFlags = [v for k, v in results.items() if isinstance(v, bool)]
        results['ensemble_detection'] = sum(detectionFlags) >= len(detectionFlags) // 2
        
        return results
        
    def trainNoveltyDetector(
        self,
        xTrain: torch.Tensor,
        numEpochs: int = 50,
        learningRate: float = 0.001
    ):
        """Train novelty detector"""
        optimizer = torch.optim.Adam(self.noveltyDetector.parameters(), lr=learningRate)
        lossFn = nn.MSELoss()
        
        self.noveltyDetector.train()
        for epoch in range(numEpochs):
            optimizer.zero_grad()
            reconstructed = self.noveltyDetector(xTrain)
            loss = lossFn(reconstructed, xTrain)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Novelty detector training epoch {epoch+1}/{numEpochs}, loss={loss.item():.6f}")
                
        # Fit threshold
        self.noveltyDetector.fitThreshold(xTrain)
