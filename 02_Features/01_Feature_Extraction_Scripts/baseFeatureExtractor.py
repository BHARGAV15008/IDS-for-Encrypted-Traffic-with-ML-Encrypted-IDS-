"""
Base Feature Extractor Module
Extracts statistical and flow-based features from network traffic data
Supports both encrypted and unencrypted traffic analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extraction"""
    
    def __init__(self, featureName: str):
        self.featureName = featureName
        self.extractedFeatures = {}
        
    @abstractmethod
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        with open(r"e:\IDS for Encrypted Traffic with ML (Encrypted IDS)\04_Source_Code\debug_shapes.log", "a") as f:
            f.write(f"Inside StatisticalFeatureExtractor - received data.shape: {data.shape}\n")
        """Extract features from input data"""
        pass
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        if data is None or data.empty:
            logger.warning(f"{self.featureName}: Empty data provided")
            return False
        return True


class StatisticalFeatureExtractor(BaseFeatureExtractor):
    """Extract statistical features from network flows"""
    
    def __init__(self):
        super().__init__("StatisticalFeatureExtractor")
        
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract statistical features including:
        - Packet size statistics (mean, std, min, max, variance)
        - Inter-arrival time statistics
        - Flow duration metrics
        - Packet count features
        """
        if not self.validate(data):
            return pd.DataFrame()
            
        features = pd.DataFrame()
        
        try:
            # Packet size statistics
            if 'packetSize' in data.columns or 'pktLen' in data.columns:
                sizeCol = 'packetSize' if 'packetSize' in data.columns else 'pktLen'
                features['pktSizeMean'] = data.groupby(data.index)[sizeCol].mean()
                features['pktSizeStd'] = data.groupby(data.index)[sizeCol].std()
                features['pktSizeMin'] = data.groupby(data.index)[sizeCol].min()
                features['pktSizeMax'] = data.groupby(data.index)[sizeCol].max()
                features['pktSizeVar'] = data.groupby(data.index)[sizeCol].var()
                
            # Inter-arrival time statistics
            if 'timestamp' in data.columns:
                iatStats = data.groupby(data.index)['timestamp'].apply(
                    lambda x: pd.Series({
                        'iatMean': x.diff().mean(),
                        'iatStd': x.diff().std(),
                        'iatMin': x.diff().min(),
                        'iatMax': x.diff().max()
                    })
                )
                features = pd.concat([features, iatStats], axis=1)
                    
            # Flow duration
            if 'timestamp' in data.columns:
                flowDuration = data.groupby(data.index)['timestamp'].apply(
                    lambda x: x.max() - x.min()
                )
                features['flowDuration'] = flowDuration
                
            # Packet counts
            features['pktCount'] = data.groupby(data.index).size()
                
            # Forward/Backward packet statistics
            if 'direction' in data.columns:
                fwdData = data[data['direction'] == 'forward']
                bwdData = data[data['direction'] == 'backward']
                
                features['fwdPktCount'] = fwdData.groupby(fwdData.index).size()
                features['bwdPktCount'] = bwdData.groupby(bwdData.index).size()
                    
            # Fill NaN values
            features = features.fillna(0)
            
            logger.info(f"Extracted {len(features.columns)} statistical features")
            
        except Exception as e:
            logger.error(f"Error extracting statistical features: {str(e)}")
            
        return features


class FlowFeatureExtractor(BaseFeatureExtractor):
    """Extract flow-level features from network traffic"""
    
    def __init__(self):
        super().__init__("FlowFeatureExtractor")
        
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract flow-level features:
        - Bytes per second
        - Packets per second
        - Flow rate metrics
        - Protocol distribution
        """
        if not self.validate(data):
            return pd.DataFrame()
            
        features = pd.DataFrame()
        
        try:
            # Bytes per second
            if 'bytes' in data.columns and 'timestamp' in data.columns:
                flowStats = data.groupby(data.index).apply(
                    lambda x: pd.Series({
                        'bytesPerSec': x['bytes'].sum() / max((x['timestamp'].max() - x['timestamp'].min()), 1e-6),
                        'totalBytes': x['bytes'].sum()
                    })
                )
                features = pd.concat([features, flowStats], axis=1)
                
            # Packets per second
            if 'timestamp' in data.columns:
                pktPerSec = data.groupby(data.index).apply(
                    lambda x: len(x) / max((x['timestamp'].max() - x['timestamp'].min()), 1e-6)
                )
                features['pktsPerSec'] = pktPerSec
                
            # Protocol distribution
            if 'protocol' in data.columns:
                protocolDist = data.groupby(data.index)['protocol'].apply(
                    lambda x: x.value_counts().to_dict()
                )
                # One-hot encode top protocols
                topProtocols = ['TCP', 'UDP', 'ICMP', 'TLS', 'HTTPS']
                for proto in topProtocols:
                    features[f'proto_{proto}'] = protocolDist.apply(
                        lambda x: x.get(proto, 0) if isinstance(x, dict) else 0
                    )
                    
            # Flow flags and characteristics
            if 'flags' in data.columns:
                flagStats = data.groupby(data.index)['flags'].apply(
                    lambda x: pd.Series({
                        'synCount': x.str.contains('SYN', na=False).sum(),
                        'ackCount': x.str.contains('ACK', na=False).sum(),
                        'finCount': x.str.contains('FIN', na=False).sum(),
                        'rstCount': x.str.contains('RST', na=False).sum()
                    })
                )
                features = pd.concat([features, flagStats], axis=1)
                
            features = features.fillna(0)
            logger.info(f"Extracted {len(features.columns)} flow features")
            
        except Exception as e:
            logger.error(f"Error extracting flow features: {str(e)}")
            
        return features


class EncryptedTrafficFeatureExtractor(BaseFeatureExtractor):
    """Extract features specific to encrypted traffic (TLS/SSL)"""
    
    def __init__(self):
        super().__init__("EncryptedTrafficFeatureExtractor")
        
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract encrypted traffic features:
        - TLS handshake characteristics
        - Certificate features
        - Cipher suite information
        - Encrypted payload patterns
        """
        if not self.validate(data):
            return pd.DataFrame()
            
        features = pd.DataFrame()
        
        try:
            # TLS version distribution
            if 'tlsVersion' in data.columns:
                tlsVersions = data.groupby(data.index)['tlsVersion'].apply(
                    lambda x: pd.Series({
                        'tlsV1_0': (x == 'TLSv1.0').sum(),
                        'tlsV1_1': (x == 'TLSv1.1').sum(),
                        'tlsV1_2': (x == 'TLSv1.2').sum(),
                        'tlsV1_3': (x == 'TLSv1.3').sum()
                    })
                )
                features = pd.concat([features, tlsVersions], axis=1)
                
            # Handshake message counts
            if 'tlsHandshakeType' in data.columns:
                handshakeStats = data.groupby(data.index)['tlsHandshakeType'].apply(
                    lambda x: pd.Series({
                        'clientHelloCount': (x == 'ClientHello').sum(),
                        'serverHelloCount': (x == 'ServerHello').sum(),
                        'certificateCount': (x == 'Certificate').sum(),
                        'keyExchangeCount': (x == 'KeyExchange').sum()
                    })
                )
                features = pd.concat([features, handshakeStats], axis=1)
                
            # Certificate features
            if 'certLength' in data.columns:
                certStats = data.groupby(data.index)['certLength'].agg(['mean', 'max', 'count'])
                certStats.columns = ['certLengthMean', 'certLengthMax', 'certCount']
                features = pd.concat([features, certStats], axis=1)
                
            # Cipher suite complexity
            if 'cipherSuite' in data.columns:
                cipherStats = data.groupby(data.index)['cipherSuite'].apply(
                    lambda x: pd.Series({
                        'uniqueCiphers': x.nunique(),
                        'hasAES': x.str.contains('AES', na=False).any(),
                        'hasGCM': x.str.contains('GCM', na=False).any(),
                        'hasECDHE': x.str.contains('ECDHE', na=False).any()
                    })
                )
                features = pd.concat([features, cipherStats], axis=1)
                
            # Encrypted payload size patterns
            if 'payloadSize' in data.columns:
                payloadStats = data.groupby(data.index)['payloadSize'].apply(
                    lambda x: pd.Series({
                        'payloadSizeMean': x.mean(),
                        'payloadSizeStd': x.std(),
                        'payloadSizeEntropy': self._calculateEntropy(x.values)
                    })
                )
                features = pd.concat([features, payloadStats], axis=1)
                
            features = features.fillna(0)
            logger.info(f"Extracted {len(features.columns)} encrypted traffic features")
            
        except Exception as e:
            logger.error(f"Error extracting encrypted traffic features: {str(e)}")
            
        return features
    
    def _calculateEntropy(self, values: np.ndarray) -> float:
        """Calculate Shannon entropy of value distribution"""
        if len(values) == 0:
            return 0.0
        valueCounts = pd.Series(values).value_counts(normalize=True)
        entropy = -np.sum(valueCounts * np.log2(valueCounts + 1e-10))
        return entropy


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """Extract temporal and sequential features from traffic flows"""
    
    def __init__(self, windowSize: int = 10):
        super().__init__("TemporalFeatureExtractor")
        self.windowSize = windowSize
        
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features:
        - Sliding window statistics
        - Burst detection
        - Periodicity measures
        - Time-series patterns
        """
        if not self.validate(data):
            return pd.DataFrame()
            
        features = pd.DataFrame()
        
        try:
            if 'timestamp' not in data.columns:
                logger.warning("No timestamp column found")
                return features
                
            # Sort by timestamp
            data = data.sort_values([data.index.name, 'timestamp'])
            
            # Sliding window statistics
            for flowId, flowData in data.groupby(data.index):
                if len(flowData) < self.windowSize:
                    continue
                    
                # Calculate rolling statistics
                if 'packetSize' in flowData.columns or 'pktLen' in flowData.columns:
                    sizeCol = 'packetSize' if 'packetSize' in flowData.columns else 'pktLen'
                    rolling = flowData[sizeCol].rolling(window=self.windowSize)
                    
                    flowFeatures = pd.Series({
                        'rollingMean': rolling.mean().mean(),
                        'rollingStd': rolling.std().mean(),
                        'rollingMax': rolling.max().max(),
                        'rollingMin': rolling.min().min()
                    }, name=flowId)
                    
                    features = pd.concat([features, flowFeatures.to_frame().T])
                    
            # Burst detection
            burstFeatures = data.groupby(data.index).apply(self._detectBursts)
            features = pd.concat([features, burstFeatures], axis=1)
            
            # Periodicity measures
            periodicityFeatures = data.groupby(data.index).apply(self._calculatePeriodicity)
            features = pd.concat([features, periodicityFeatures], axis=1)
            
            features = features.fillna(0)
            logger.info(f"Extracted {len(features.columns)} temporal features")
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {str(e)}")
            
        return features
    
    def _detectBursts(self, flowData: pd.DataFrame) -> pd.Series:
        """Detect traffic bursts in flow"""
        if 'timestamp' not in flowData.columns or len(flowData) < 2:
            return pd.Series({'burstCount': 0, 'avgBurstSize': 0})
            
        iat = flowData['timestamp'].diff()
        threshold = iat.mean() - iat.std()
        
        bursts = (iat < threshold).sum()
        avgBurstSize = len(flowData) / max(bursts, 1)
        
        return pd.Series({
            'burstCount': bursts,
            'avgBurstSize': avgBurstSize
        })
    
    def _calculatePeriodicity(self, flowData: pd.DataFrame) -> pd.Series:
        """Calculate periodicity measures using autocorrelation"""
        if 'timestamp' not in flowData.columns or len(flowData) < 10:
            return pd.Series({'periodicity': 0, 'periodicityStrength': 0})
            
        try:
            iat = flowData['timestamp'].diff().dropna()
            if len(iat) < 10:
                return pd.Series({'periodicity': 0, 'periodicityStrength': 0})
                
            # Simple autocorrelation at lag 1
            autocorr = iat.autocorr(lag=1)
            
            return pd.Series({
                'periodicity': autocorr if not np.isnan(autocorr) else 0,
                'periodicityStrength': abs(autocorr) if not np.isnan(autocorr) else 0
            })
        except:
            return pd.Series({'periodicity': 0, 'periodicityStrength': 0})


class FeatureExtractionPipeline:
    """Orchestrates multiple feature extractors"""
    
    def __init__(self, extractors: Optional[List[BaseFeatureExtractor]] = None):
        if extractors is None:
            self.extractors = [
                StatisticalFeatureExtractor(),
                EncryptedTrafficFeatureExtractor(),
            ]
        else:
            self.extractors = extractors
            
    def extractAll(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from data"""
        with open(r"e:\IDS for Encrypted Traffic with ML (Encrypted IDS)\04_Source_Code\debug_shapes.log", "a") as f:
            f.write(f"Inside FeatureExtractionPipeline - received data.shape: {data.shape}\n")
        allFeatures = []
        
        for extractor in self.extractors:
            logger.info(f"Running {extractor.featureName}...")
            features = extractor.extract(data)
            logger.info(f"  - Extracted features shape: {features.shape}")
            with open(r"e:\IDS for Encrypted Traffic with ML (Encrypted IDS)\04_Source_Code\debug_shapes.log", "a") as f:
                f.write(f"Inside FeatureExtractionPipeline - After {extractor.featureName} - features.shape: {features.shape}\n")
            if not features.empty:
                allFeatures.append(features)
                
        if not allFeatures:
            logger.warning("No features extracted")
            return pd.DataFrame()
            
        # Combine all features
        logger.info(f"Combining {len(allFeatures)} feature dataframes.")
        combinedFeatures = pd.concat(allFeatures, axis=1)
        logger.info(f"  - Combined features shape: {combinedFeatures.shape}")
        
        # Remove duplicate columns
        combinedFeatures = combinedFeatures.loc[:, ~combinedFeatures.columns.duplicated()]
        
        logger.info(f"Total features extracted: {len(combinedFeatures.columns)}")
        
        return combinedFeatures
