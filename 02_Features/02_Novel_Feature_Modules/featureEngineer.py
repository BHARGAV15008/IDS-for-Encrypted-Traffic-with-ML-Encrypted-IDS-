"""
Feature Engineering Module
Implements advanced feature engineering techniques including:
- Feature transformation
- Feature interaction
- Domain-specific feature creation
- Invariance-driven features for adversarial robustness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
import logging

logger = logging.getLogger(__name__)


class FeatureTransformer:
    """Transform features using various techniques"""
    
    def __init__(self, method: str = 'standard'):
        """
        Args:
            method: Transformation method ('standard', 'minmax', 'robust', 'log', 'sqrt')
        """
        self.method = method
        self.scaler = None
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
            
    def fit(self, features: pd.DataFrame) -> 'FeatureTransformer':
        """Fit transformer on features"""
        if self.scaler is not None:
            self.scaler.fit(features)
        return self
        
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features"""
        if self.method in ['standard', 'minmax', 'robust']:
            transformed = self.scaler.transform(features)
            return pd.DataFrame(transformed, columns=features.columns, index=features.index)
            
        elif self.method == 'log':
            return np.log1p(features.abs())
            
        elif self.method == 'sqrt':
            return np.sqrt(features.abs())
            
        else:
            return features
            
    def fitTransform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features"""
        return self.fit(features).transform(features)


class FeatureInteractionGenerator:
    """Generate interaction features between existing features"""
    
    def __init__(self, degree: int = 2, interactionOnly: bool = True):
        """
        Args:
            degree: Polynomial degree for interactions
            interactionOnly: If True, only create interaction terms (no powers)
        """
        self.degree = degree
        self.interactionOnly = interactionOnly
        self.polyFeatures = PolynomialFeatures(
            degree=degree,
            interaction_only=interactionOnly,
            include_bias=False
        )
        
    def generate(self, features: pd.DataFrame, maxFeatures: int = 100) -> pd.DataFrame:
        """
        Generate interaction features
        
        Args:
            features: Input features
            maxFeatures: Maximum number of features to select (to avoid explosion)
        """
        try:
            # Select most important features for interaction
            if len(features.columns) > 20:
                # Use variance as simple importance measure
                variances = features.var()
                topFeatures = variances.nlargest(20).index
                features = features[topFeatures]
                
            # Generate polynomial features
            interactionFeatures = self.polyFeatures.fit_transform(features)
            
            # Get feature names
            featureNames = self.polyFeatures.get_feature_names_out(features.columns)
            
            # Create DataFrame
            interactionDf = pd.DataFrame(
                interactionFeatures,
                columns=featureNames,
                index=features.index
            )
            
            # Remove original features (keep only interactions)
            if self.interactionOnly:
                interactionDf = interactionDf.drop(columns=features.columns, errors='ignore')
                
            # Limit number of features
            if len(interactionDf.columns) > maxFeatures:
                variances = interactionDf.var()
                topCols = variances.nlargest(maxFeatures).index
                interactionDf = interactionDf[topCols]
                
            logger.info(f"Generated {len(interactionDf.columns)} interaction features")
            
            return interactionDf
            
        except Exception as e:
            logger.error(f"Error generating interaction features: {str(e)}")
            return pd.DataFrame()


class InvarianceFeatureEngineer:
    """
    Create invariance-driven features for adversarial robustness
    These features are designed to be resistant to small perturbations
    """
    
    def __init__(self):
        self.featureName = "InvarianceFeatureEngineer"
        
    def engineer(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Create invariance-driven features:
        - Ratio-based features (scale-invariant)
        - Rank-based features (order-preserving)
        - Binned features (discretized)
        - Aggregated features (smoothed)
        """
        invariantFeatures = pd.DataFrame(index=features.index)
        
        try:
            # Ratio-based features (scale-invariant)
            ratioFeatures = self._createRatioFeatures(features)
            invariantFeatures = pd.concat([invariantFeatures, ratioFeatures], axis=1)
            
            # Rank-based features (order-preserving)
            rankFeatures = self._createRankFeatures(features)
            invariantFeatures = pd.concat([invariantFeatures, rankFeatures], axis=1)
            
            # Binned features (discretized)
            binnedFeatures = self._createBinnedFeatures(features)
            invariantFeatures = pd.concat([invariantFeatures, binnedFeatures], axis=1)
            
            # Aggregated features (smoothed)
            aggregatedFeatures = self._createAggregatedFeatures(features)
            invariantFeatures = pd.concat([invariantFeatures, aggregatedFeatures], axis=1)
            
            logger.info(f"Created {len(invariantFeatures.columns)} invariance-driven features")
            
        except Exception as e:
            logger.error(f"Error creating invariance features: {str(e)}")
            
        return invariantFeatures
    
    def _createRatioFeatures(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create ratio-based features"""
        ratios = pd.DataFrame(index=features.index)
        
        # Find numeric columns
        numericCols = features.select_dtypes(include=[np.number]).columns
        
        # Create ratios for related features
        featurePairs = [
            ('pktSizeMean', 'pktSizeStd'),
            ('fwdPktCount', 'bwdPktCount'),
            ('iatMean', 'iatStd'),
            ('totalBytes', 'pktCount'),
            ('flowDuration', 'pktCount')
        ]
        
        for feat1, feat2 in featurePairs:
            if feat1 in numericCols and feat2 in numericCols:
                # Ensure that the denominator is not zero to avoid division by zero errors
                if features[feat2].abs().sum() > 1e-6:
                    ratioName = f'ratio_{feat1}_{feat2}'
                    ratios[ratioName] = features[feat1] / (features[feat2] + 1e-10)
                else:
                    ratioName = f'ratio_{feat1}_{feat2}'
                    ratios[ratioName] = 0
                
        return ratios.fillna(0).replace([np.inf, -np.inf], 0)
    
    def _createRankFeatures(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create rank-based features"""
        ranks = pd.DataFrame(index=features.index)
        
        # Rank features (percentile-based)
        numericCols = features.select_dtypes(include=[np.number]).columns[:10]  # Limit to top 10
        
        for col in numericCols:
            rankName = f'rank_{col}'
            ranks[rankName] = features[col].rank(pct=True)
            
        return ranks.fillna(0)
    
    def _createBinnedFeatures(self, features: pd.DataFrame, numBins: int = 5) -> pd.DataFrame:
        """Create binned/discretized features"""
        binned = pd.DataFrame(index=features.index)
        
        numericCols = features.select_dtypes(include=[np.number]).columns[:10]
        
        for col in numericCols:
            try:
                binnedName = f'binned_{col}'
                binned[binnedName] = pd.qcut(
                    features[col],
                    q=numBins,
                    labels=False,
                    duplicates='drop'
                )
            except:
                # If qcut fails, use cut
                try:
                    binned[binnedName] = pd.cut(
                        features[col],
                        bins=numBins,
                        labels=False
                    )
                except:
                    pass
                    
        return binned.fillna(0)
    
    def _createAggregatedFeatures(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated/smoothed features"""
        aggregated = pd.DataFrame(index=features.index)
        
        numericCols = features.select_dtypes(include=[np.number]).columns
        
        # Group similar features and aggregate
        featureGroups = {
            'pktSize': [c for c in numericCols if 'pktsize' in c.lower() or 'packetsize' in c.lower()],
            'iat': [c for c in numericCols if 'iat' in c.lower()],
            'flow': [c for c in numericCols if 'flow' in c.lower()],
            'entropy': [c for c in numericCols if 'entropy' in c.lower()]
        }
        
        for groupName, groupCols in featureGroups.items():
            if len(groupCols) > 0:
                # Ensure that the operations are performed on the correct axis and preserve the index
                aggregated[f'{groupName}_mean'] = features[groupCols].mean(axis=1)
                aggregated[f'{groupName}_max'] = features[groupCols].max(axis=1)
                aggregated[f'{groupName}_min'] = features[groupCols].min(axis=1)
            else:
                # If no columns are found for a group, fill with 0 to maintain consistency
                aggregated[f'{groupName}_mean'] = 0
                aggregated[f'{groupName}_max'] = 0
                aggregated[f'{groupName}_min'] = 0
                
        return aggregated.fillna(0)


class DomainSpecificFeatureEngineer:
    """Create domain-specific features for network intrusion detection"""
    
    def __init__(self):
        self.featureName = "DomainSpecificFeatureEngineer"
        
    def engineer(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features:
        - Attack signature features
        - Protocol anomaly features
        - Timing attack features
        - Encryption quality features
        """
        domainFeatures = pd.DataFrame(index=features.index)
        logger.info(f"Initial shape in domain engineer: {features.shape}")
        
        try:
            # Attack signature features
            attackFeatures = self._createAttackSignatures(features)
            logger.info(f"Shape after attack signatures: {attackFeatures.shape}")
            domainFeatures = pd.concat([domainFeatures, attackFeatures], axis=1)
            
            # Protocol anomaly features
            protocolFeatures = self._createProtocolAnomalies(features)
            logger.info(f"Shape after protocol anomalies: {protocolFeatures.shape}")
            domainFeatures = pd.concat([domainFeatures, protocolFeatures], axis=1)
            
            # Timing attack features
            timingFeatures = self._createTimingFeatures(features)
            logger.info(f"Shape after timing features: {timingFeatures.shape}")
            domainFeatures = pd.concat([domainFeatures, timingFeatures], axis=1)
            
            # Encryption quality features
            encryptionFeatures = self._createEncryptionFeatures(features)
            logger.info(f"Shape after encryption features: {encryptionFeatures.shape}")
            domainFeatures = pd.concat([domainFeatures, encryptionFeatures], axis=1)
            
            logger.info(f"Final shape in domain engineer: {domainFeatures.shape}")
            logger.info(f"Created {len(domainFeatures.columns)} domain-specific features")
            
        except Exception as e:
            logger.error(f"Error creating domain-specific features: {str(e)}")
            
        return domainFeatures
    
    def _createAttackSignatures(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create features indicative of specific attack types"""
        signatures = pd.DataFrame(index=features.index)
        
        # Port scan signature
        if 'uniquePorts' in features.columns:
            signatures['portScanSignature'] = (features['uniquePorts'] > 10).astype(int)
            
        # DDoS signature
        if 'pktsPerSec' in features.columns and 'pktCount' in features.columns:
            signatures['ddosSignature'] = (
                (features['pktsPerSec'] > features['pktsPerSec'].quantile(0.95)) &
                (features['pktCount'] > features['pktCount'].quantile(0.9))
            ).astype(int)
            
        # Data exfiltration signature
        if 'totalBytes' in features.columns and 'flowDuration' in features.columns:
            signatures['exfiltrationSignature'] = (
                (features['totalBytes'] > features['totalBytes'].quantile(0.9)) &
                (features['flowDuration'] > features['flowDuration'].quantile(0.8))
            ).astype(int)
            
        # Brute force signature
        if 'fwdBwdRatio' in features.columns and 'pktCount' in features.columns:
            signatures['bruteForceSignature'] = (
                (features['fwdBwdRatio'] > 2) &
                (features['pktCount'] > features['pktCount'].median())
            ).astype(int)
            
        return signatures.fillna(0)
    
    def _createProtocolAnomalies(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create features detecting protocol anomalies"""
        anomalies = pd.DataFrame(index=features.index)
        
        # Unusual packet size for protocol
        if 'pktSizeMean' in features.columns:
            anomalies['unusualPktSize'] = (
                (features['pktSizeMean'] < features['pktSizeMean'].quantile(0.05)) |
                (features['pktSizeMean'] > features['pktSizeMean'].quantile(0.95))
            ).astype(int)
            
        # Unusual inter-arrival time
        if 'iatMean' in features.columns:
            anomalies['unusualIAT'] = (
                (features['iatMean'] < features['iatMean'].quantile(0.05)) |
                (features['iatMean'] > features['iatMean'].quantile(0.95))
            ).astype(int)
            
        # Protocol entropy anomaly
        if 'protocolEntropy' in features.columns:
            anomalies['protocolEntropyAnomaly'] = (
                features['protocolEntropy'] > features['protocolEntropy'].quantile(0.9)
            ).astype(int)
            
        return anomalies.fillna(0)
    
    def _createTimingFeatures(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create timing-based attack detection features"""
        timing = pd.DataFrame(index=features.index)
        
        # Regularity score (low variance in IAT suggests automation)
        if 'iatStd' in features.columns and 'iatMean' in features.columns:
            timing['timingRegularity'] = features['iatStd'] / (features['iatMean'] + 1e-10)
            
        # Burst intensity
        if 'burstCount' in features.columns and 'flowDuration' in features.columns:
            timing['burstIntensity'] = features['burstCount'] / (features['flowDuration'] + 1e-10)
            
        # Activity concentration
        if 'activityIntensity' in features.columns:
            timing['activityConcentration'] = (
                features['activityIntensity'] > features['activityIntensity'].quantile(0.9)
            ).astype(int)
            
        return timing.fillna(0).replace([np.inf, -np.inf], 0)
    
    def _createEncryptionFeatures(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create features related to encryption quality and anomalies"""
        encryption = pd.DataFrame(index=features.index)
        
        # TLS version score (higher = more secure)
        if 'tlsV1_3' in features.columns:
            encryption['tlsVersionScore'] = (
                features.get('tlsV1_3', 0) * 4 +
                features.get('tlsV1_2', 0) * 3 +
                features.get('tlsV1_1', 0) * 2 +
                features.get('tlsV1_0', 0) * 1
            )
            
        # Cipher strength indicator
        if 'hasAES' in features.columns and 'hasGCM' in features.columns:
            encryption['cipherStrength'] = (
                features.get('hasAES', 0) +
                features.get('hasGCM', 0) +
                features.get('hasECDHE', 0)
            )
            
        # Handshake completeness
        if 'clientHelloCount' in features.columns and 'serverHelloCount' in features.columns:
            encryption['handshakeCompleteness'] = (
                (features['clientHelloCount'] > 0) &
                (features['serverHelloCount'] > 0)
            ).astype(int)
            
        # Payload entropy (high entropy suggests encryption)
        if 'payloadSizeEntropy' in features.columns:
            encryption['highEntropyPayload'] = (
                features['payloadSizeEntropy'] > features['payloadSizeEntropy'].quantile(0.75)
            ).astype(int)
            
        return encryption.fillna(0)


class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline"""
    
    def __init__(
        self,
        transformMethod: str = 'robust',
        createInteractions: bool = True,
        createInvariant: bool = True,
        createDomainSpecific: bool = True
    ):
        self.transformMethod = transformMethod
        self.createInteractions = createInteractions
        self.createInvariant = createInvariant
        self.createDomainSpecific = createDomainSpecific
        
        self.transformer = FeatureTransformer(method=transformMethod)
        self.interactionGenerator = FeatureInteractionGenerator()
        self.invarianceEngineer = InvarianceFeatureEngineer()
        self.domainEngineer = DomainSpecificFeatureEngineer()
        
    def fit(self, features: pd.DataFrame) -> 'FeatureEngineeringPipeline':
        """Fit the pipeline on features"""
        # Create all features first to fit the transformer correctly
        engineeredFeatures = features.copy()

        # Create invariance-driven features
        if self.createInvariant:
            logger.info(f"Shape before invariance features: {engineeredFeatures.shape}")
            invariantFeatures = self.invarianceEngineer.engineer(features)
            logger.info(f"Shape after invariance features: {invariantFeatures.shape}")
            engineeredFeatures = pd.concat([engineeredFeatures, invariantFeatures], axis=1)
            logger.info(f"Shape after concat invariance: {engineeredFeatures.shape}")

        # Create domain-specific features
        if self.createDomainSpecific:
            logger.info(f"Shape before domain features: {engineeredFeatures.shape}")
            domainFeatures = self.domainEngineer.engineer(features)
            logger.info(f"Shape after domain features: {domainFeatures.shape}")
            engineeredFeatures = pd.concat([engineeredFeatures, domainFeatures], axis=1)
            logger.info(f"Shape after concat domain: {engineeredFeatures.shape}")

        # Create interaction features
        if self.createInteractions:
            logger.info(f"Shape before interaction features: {engineeredFeatures.shape}")
            interactionFeatures = self.interactionGenerator.generate(features)
            logger.info(f"Shape after interaction features: {interactionFeatures.shape}")
            engineeredFeatures = pd.concat([engineeredFeatures, interactionFeatures], axis=1)
            logger.info(f"Shape after concat interaction: {engineeredFeatures.shape}")
            
        # Remove duplicates before fitting
        engineeredFeatures = engineeredFeatures.loc[:, ~engineeredFeatures.columns.duplicated()]

        # Now fit the transformer on the complete set of features
        numericCols = engineeredFeatures.select_dtypes(include=[np.number]).columns
        self.transformer.fit(engineeredFeatures[numericCols])
        return self
        
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features through the pipeline"""
        engineeredFeatures = features.copy()
        print(f"Initial shape in transform: {engineeredFeatures.shape}", flush=True)
        
        # Create invariance-driven features
        if self.createInvariant:
            print(f"Shape before invariance transform: {engineeredFeatures.shape}", flush=True)
            invariantFeatures = self.invarianceEngineer.engineer(features)
            print(f"Shape of invariantFeatures: {invariantFeatures.shape}", flush=True)
            engineeredFeatures = pd.concat([engineeredFeatures, invariantFeatures], axis=1)
            print(f"Shape after adding invariantFeatures: {engineeredFeatures.shape}", flush=True)
            
        # Create domain-specific features
        if self.createDomainSpecific:
            print(f"Shape before domain transform: {engineeredFeatures.shape}", flush=True)
            domainFeatures = self.domainEngineer.engineer(features)
            print(f"Shape of domainFeatures: {domainFeatures.shape}", flush=True)
            engineeredFeatures = pd.concat([engineeredFeatures, domainFeatures], axis=1)
            print(f"Shape after adding domainFeatures: {engineeredFeatures.shape}", flush=True)
            
        # Create interaction features
        if self.createInteractions:
            print(f"Shape before interaction transform: {engineeredFeatures.shape}", flush=True)
            interactionFeatures = self.interactionGenerator.generate(features)
            print(f"Shape of interactionFeatures: {interactionFeatures.shape}", flush=True)
            engineeredFeatures = pd.concat([engineeredFeatures, interactionFeatures], axis=1)
            print(f"Shape after adding interactionFeatures: {engineeredFeatures.shape}", flush=True)
            
        # Transform features
        numericCols = engineeredFeatures.select_dtypes(include=[np.number]).columns
        engineeredFeatures[numericCols] = self.transformer.transform(engineeredFeatures[numericCols])
        
        # Remove duplicates
        engineeredFeatures = engineeredFeatures.loc[:, ~engineeredFeatures.columns.duplicated()]
        
        # Final check for any NaNs/Infs that may have been created
        engineeredFeatures.replace([np.inf, -np.inf], np.nan, inplace=True)
        engineeredFeatures.fillna(0, inplace=True)
        
        print(f"Feature engineering complete: {len(engineeredFeatures.columns)} total features", flush=True)
        print(f"Final shape after transform: {engineeredFeatures.shape}", flush=True)
        
        return engineeredFeatures
        
    def fitTransform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features"""
        return self.fit(features).transform(features)
