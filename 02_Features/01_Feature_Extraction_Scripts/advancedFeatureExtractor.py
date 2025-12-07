"""
Advanced Feature Extractor Module
Implements sophisticated feature extraction techniques for encrypted traffic analysis
Includes entropy-based, graph-based, and behavioral features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.fft import fft
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class EntropyFeatureExtractor:
    """Extract entropy-based features for anomaly detection"""
    
    def __init__(self):
        self.featureName = "EntropyFeatureExtractor"
        
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract entropy-based features:
        - Shannon entropy
        - Conditional entropy
        - Relative entropy (KL divergence)
        - Approximate entropy
        """
        features = pd.DataFrame()
        
        try:
            # Packet size entropy
            if 'packetSize' in data.columns or 'pktLen' in data.columns:
                sizeCol = 'packetSize' if 'packetSize' in data.columns else 'pktLen'
                
                entropyStats = data.groupby(data.index)[sizeCol].apply(
                    lambda x: pd.Series({
                        'pktSizeEntropy': self._shannonEntropy(x.values),
                        'pktSizeApproxEntropy': self._approximateEntropy(x.values),
                        'pktSizeNormalizedEntropy': self._normalizedEntropy(x.values)
                    })
                )
                features = pd.concat([features, entropyStats], axis=1)
                
            # Inter-arrival time entropy
            if 'timestamp' in data.columns:
                iatEntropy = data.groupby(data.index)['timestamp'].apply(
                    lambda x: pd.Series({
                        'iatEntropy': self._shannonEntropy(x.diff().dropna().values),
                        'iatApproxEntropy': self._approximateEntropy(x.diff().dropna().values)
                    })
                )
                features = pd.concat([features, iatEntropy], axis=1)
                
            # Protocol entropy
            if 'protocol' in data.columns:
                protocolEntropy = data.groupby(data.index)['protocol'].apply(
                    lambda x: self._shannonEntropy(x.values)
                )
                features['protocolEntropy'] = protocolEntropy
                
            # Port entropy
            if 'dstPort' in data.columns:
                portEntropy = data.groupby(data.index)['dstPort'].apply(
                    lambda x: self._shannonEntropy(x.values)
                )
                features['dstPortEntropy'] = portEntropy
                
            features = features.fillna(0)
            logger.info(f"Extracted {len(features.columns)} entropy features")
            
        except Exception as e:
            logger.error(f"Error extracting entropy features: {str(e)}")
            
        return features
    
    def _shannonEntropy(self, values: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        if len(values) == 0:
            return 0.0
            
        valueCounts = Counter(values)
        total = len(values)
        entropy = 0.0
        
        for count in valueCounts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * np.log2(prob)
                
        return entropy
    
    def _approximateEntropy(self, values: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate approximate entropy (ApEn)"""
        if len(values) < m + 1:
            return 0.0
            
        try:
            n = len(values)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([values[i:i + m] for i in range(n - m + 1)])
                c = np.zeros(n - m + 1)
                
                for i in range(n - m + 1):
                    template = patterns[i]
                    for j in range(n - m + 1):
                        if _maxdist(template, patterns[j], m) <= r * np.std(values):
                            c[i] += 1
                            
                c = c / (n - m + 1)
                return np.sum(np.log(c + 1e-10)) / (n - m + 1)
            
            return abs(_phi(m) - _phi(m + 1))
            
        except:
            return 0.0
    
    def _normalizedEntropy(self, values: np.ndarray) -> float:
        """Calculate normalized entropy (0-1 range)"""
        entropy = self._shannonEntropy(values)
        maxEntropy = np.log2(len(np.unique(values)) + 1e-10)
        return entropy / maxEntropy if maxEntropy > 0 else 0.0


class StatisticalMomentExtractor:
    """Extract statistical moments and distribution features"""
    
    def __init__(self):
        self.featureName = "StatisticalMomentExtractor"
        
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract statistical moments:
        - Skewness
        - Kurtosis
        - Coefficient of variation
        - Percentiles
        """
        features = pd.DataFrame()
        
        try:
            # Packet size moments
            if 'packetSize' in data.columns or 'pktLen' in data.columns:
                sizeCol = 'packetSize' if 'packetSize' in data.columns else 'pktLen'
                
                momentStats = data.groupby(data.index)[sizeCol].apply(
                    lambda x: pd.Series({
                        'pktSizeSkewness': stats.skew(x),
                        'pktSizeKurtosis': stats.kurtosis(x),
                        'pktSizeCV': x.std() / (x.mean() + 1e-10),
                        'pktSize25Percentile': np.percentile(x, 25),
                        'pktSize50Percentile': np.percentile(x, 50),
                        'pktSize75Percentile': np.percentile(x, 75),
                        'pktSize90Percentile': np.percentile(x, 90),
                        'pktSizeIQR': np.percentile(x, 75) - np.percentile(x, 25)
                    })
                )
                features = pd.concat([features, momentStats], axis=1)
                
            # Inter-arrival time moments
            if 'timestamp' in data.columns:
                iatMoments = data.groupby(data.index)['timestamp'].apply(
                    lambda x: pd.Series({
                        'iatSkewness': stats.skew(x.diff().dropna()),
                        'iatKurtosis': stats.kurtosis(x.diff().dropna()),
                        'iatCV': x.diff().std() / (x.diff().mean() + 1e-10)
                    })
                )
                features = pd.concat([features, iatMoments], axis=1)
                
            features = features.fillna(0)
            features = features.replace([np.inf, -np.inf], 0)
            
            logger.info(f"Extracted {len(features.columns)} statistical moment features")
            
        except Exception as e:
            logger.error(f"Error extracting statistical moments: {str(e)}")
            
        return features


class FrequencyDomainExtractor:
    """Extract frequency domain features using FFT"""
    
    def __init__(self, numComponents: int = 10):
        self.featureName = "FrequencyDomainExtractor"
        self.numComponents = numComponents
        
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract frequency domain features:
        - FFT coefficients
        - Spectral energy
        - Dominant frequencies
        """
        features = pd.DataFrame()
        
        try:
            # FFT of packet sizes
            if 'packetSize' in data.columns or 'pktLen' in data.columns:
                sizeCol = 'packetSize' if 'packetSize' in data.columns else 'pktLen'
                
                fftFeatures = data.groupby(data.index)[sizeCol].apply(
                    lambda x: self._extractFFTFeatures(x.values)
                )
                
                # Expand FFT features into separate columns
                fftDf = pd.DataFrame(fftFeatures.tolist(), index=fftFeatures.index)
                fftDf.columns = [f'fftCoeff_{i}' for i in range(fftDf.shape[1])]
                features = pd.concat([features, fftDf], axis=1)
                
            # FFT of inter-arrival times
            if 'timestamp' in data.columns:
                iatFFT = data.groupby(data.index)['timestamp'].apply(
                    lambda x: self._extractFFTFeatures(x.diff().dropna().values)
                )
                
                iatFftDf = pd.DataFrame(iatFFT.tolist(), index=iatFFT.index)
                iatFftDf.columns = [f'iatFFTCoeff_{i}' for i in range(iatFftDf.shape[1])]
                features = pd.concat([features, iatFftDf], axis=1)
                
            features = features.fillna(0)
            logger.info(f"Extracted {len(features.columns)} frequency domain features")
            
        except Exception as e:
            logger.error(f"Error extracting frequency domain features: {str(e)}")
            
        return features
    
    def _extractFFTFeatures(self, signal: np.ndarray) -> np.ndarray:
        """Extract FFT features from signal"""
        if len(signal) < 2:
            return np.zeros(self.numComponents)
            
        try:
            # Pad or truncate signal
            if len(signal) < self.numComponents * 2:
                signal = np.pad(signal, (0, self.numComponents * 2 - len(signal)), 'constant')
            else:
                signal = signal[:self.numComponents * 2]
                
            # Compute FFT
            fftCoeffs = fft(signal)
            
            # Take magnitude of first numComponents coefficients
            magnitudes = np.abs(fftCoeffs[:self.numComponents])
            
            # Normalize
            magnitudes = magnitudes / (np.max(magnitudes) + 1e-10)
            
            return magnitudes
            
        except:
            return np.zeros(self.numComponents)


class BehavioralFeatureExtractor:
    """Extract behavioral features from traffic patterns"""
    
    def __init__(self):
        self.featureName = "BehavioralFeatureExtractor"
        
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract behavioral features:
        - Connection patterns
        - Request/response ratios
        - Session characteristics
        - Anomalous behavior indicators
        """
        features = pd.DataFrame()
        
        try:
            # Connection patterns
            if 'srcIP' in data.columns and 'dstIP' in data.columns:
                connectionStats = data.groupby(data.index).apply(
                    lambda x: pd.Series({
                        'uniqueSrcIPs': x['srcIP'].nunique(),
                        'uniqueDstIPs': x['dstIP'].nunique(),
                        'ipDiversity': x['srcIP'].nunique() + x['dstIP'].nunique()
                    })
                )
                features = pd.concat([features, connectionStats], axis=1)
                
            # Port scanning indicators
            if 'dstPort' in data.columns:
                portStats = data.groupby(data.index)['dstPort'].apply(
                    lambda x: pd.Series({
                        'uniquePorts': x.nunique(),
                        'portScanIndicator': 1 if x.nunique() > 10 else 0,
                        'portRange': x.max() - x.min() if len(x) > 0 else 0
                    })
                )
                features = pd.concat([features, portStats], axis=1)
                
            # Request/Response patterns
            if 'direction' in data.columns:
                directionStats = data.groupby(data.index)['direction'].apply(
                    lambda x: pd.Series({
                        'fwdBwdRatio': (x == 'forward').sum() / max((x == 'backward').sum(), 1),
                        'directionChanges': (x != x.shift()).sum()
                    })
                )
                features = pd.concat([features, directionStats], axis=1)
                
            # Payload size variation (potential data exfiltration)
            if 'payloadSize' in data.columns:
                payloadStats = data.groupby(data.index)['payloadSize'].apply(
                    lambda x: pd.Series({
                        'payloadVariation': x.std() / (x.mean() + 1e-10),
                        'largePayloadCount': (x > x.quantile(0.9)).sum(),
                        'smallPayloadCount': (x < x.quantile(0.1)).sum()
                    })
                )
                features = pd.concat([features, payloadStats], axis=1)
                
            # Time-based behavioral patterns
            if 'timestamp' in data.columns:
                timeStats = data.groupby(data.index)['timestamp'].apply(
                    lambda x: pd.Series({
                        'activityDuration': x.max() - x.min(),
                        'activityIntensity': len(x) / max((x.max() - x.min()), 1e-6),
                        'silencePeriods': self._detectSilencePeriods(x.values)
                    })
                )
                features = pd.concat([features, timeStats], axis=1)
                
            features = features.fillna(0)
            features = features.replace([np.inf, -np.inf], 0)
            
            logger.info(f"Extracted {len(features.columns)} behavioral features")
            
        except Exception as e:
            logger.error(f"Error extracting behavioral features: {str(e)}")
            
        return features
    
    def _detectSilencePeriods(self, timestamps: np.ndarray) -> int:
        """Detect periods of silence in traffic"""
        if len(timestamps) < 2:
            return 0
            
        diffs = np.diff(sorted(timestamps))
        threshold = np.mean(diffs) + 2 * np.std(diffs)
        
        return np.sum(diffs > threshold)


class GraphBasedFeatureExtractor:
    """Extract graph-based features from network connections"""
    
    def __init__(self):
        self.featureName = "GraphBasedFeatureExtractor"
        
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract graph-based features:
        - Node degree
        - Clustering coefficient
        - Betweenness centrality approximation
        - Community structure indicators
        """
        features = pd.DataFrame()
        
        try:
            if 'srcIP' not in data.columns or 'dstIP' not in data.columns:
                logger.warning("Source and destination IP columns required for graph features")
                return features
                
            # Build adjacency information
            graphStats = data.groupby(data.index).apply(
                lambda x: self._calculateGraphMetrics(x)
            )
            
            features = pd.DataFrame(graphStats.tolist(), index=graphStats.index)
            
            logger.info(f"Extracted {len(features.columns)} graph-based features")
            
        except Exception as e:
            logger.error(f"Error extracting graph features: {str(e)}")
            
        return features
    
    def _calculateGraphMetrics(self, flowData: pd.DataFrame) -> Dict:
        """Calculate graph metrics for a flow"""
        try:
            # Create edge list
            edges = list(zip(flowData['srcIP'], flowData['dstIP']))
            
            # Count unique nodes
            nodes = set(flowData['srcIP']) | set(flowData['dstIP'])
            numNodes = len(nodes)
            numEdges = len(edges)
            
            # Calculate degree statistics
            degreeCounts = Counter([ip for edge in edges for ip in edge])
            degrees = list(degreeCounts.values())
            
            # Calculate metrics
            metrics = {
                'graphNumNodes': numNodes,
                'graphNumEdges': numEdges,
                'graphDensity': numEdges / max((numNodes * (numNodes - 1) / 2), 1),
                'graphAvgDegree': np.mean(degrees) if degrees else 0,
                'graphMaxDegree': np.max(degrees) if degrees else 0,
                'graphDegreeStd': np.std(degrees) if degrees else 0
            }
            
            return metrics
            
        except:
            return {
                'graphNumNodes': 0,
                'graphNumEdges': 0,
                'graphDensity': 0,
                'graphAvgDegree': 0,
                'graphMaxDegree': 0,
                'graphDegreeStd': 0
            }


class AdvancedFeatureExtractionPipeline:
    """Orchestrates advanced feature extractors"""
    
    def __init__(self):
        self.extractors = [
            EntropyFeatureExtractor(),
            StatisticalMomentExtractor(),
            FrequencyDomainExtractor(),
            BehavioralFeatureExtractor(),
            GraphBasedFeatureExtractor()
        ]
        
    def extractAll(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from data"""
        with open(r"e:\IDS for Encrypted Traffic with ML (Encrypted IDS)\04_Source_Code\debug_shapes.log", "a") as f:
            f.write(f"Inside AdvancedFeatureExtractionPipeline - received data.shape: {data.shape}\n")
        allFeatures = []
        
        for extractor in self.extractors:
            logger.info(f"Running {extractor.featureName}...")
            features = extractor.extract(data)
            with open(r"e:\IDS for Encrypted Traffic with ML (Encrypted IDS)\04_Source_Code\debug_shapes.log", "a") as f:
                f.write(f"Inside AdvancedFeatureExtractionPipeline - After {extractor.featureName} - features.shape: {features.shape}\n")
            if not features.empty:
                allFeatures.append(features)
                
        if not allFeatures:
            logger.warning("No advanced features extracted")
            return pd.DataFrame()
            
        # Combine all features
        combinedFeatures = pd.concat(allFeatures, axis=1)
        
        # Remove duplicate columns
        combinedFeatures = combinedFeatures.loc[:, ~combinedFeatures.columns.duplicated()]
        
        logger.info(f"Total advanced features extracted: {len(combinedFeatures.columns)}")
        
        return combinedFeatures
