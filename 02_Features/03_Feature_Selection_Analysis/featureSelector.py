"""
Feature Selection Module
Implements various feature selection techniques to identify most relevant features
Supports filter, wrapper, and embedded methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LassoCV
import logging

logger = logging.getLogger(__name__)


class VarianceFeatureSelector:
    """Select features based on variance threshold"""
    
    def __init__(self, threshold: float = 0.01):
        """
        Args:
            threshold: Minimum variance threshold
        """
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=threshold)
        self.selectedFeatures = None
        
    def fit(self, features: pd.DataFrame, labels: Optional[pd.Series] = None) -> 'VarianceFeatureSelector':
        """Fit selector on features"""
        self.selector.fit(features)
        self.selectedFeatures = features.columns[self.selector.get_support()].tolist()
        logger.info(f"Variance selection: {len(self.selectedFeatures)} features selected")
        return self
        
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features by selecting high-variance features"""
        if self.selectedFeatures is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return features[self.selectedFeatures]
        
    def fitTransform(self, features: pd.DataFrame, labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform features"""
        return self.fit(features, labels).transform(features)


class UnivariateFeatureSelector:
    """Select features using univariate statistical tests"""
    
    def __init__(self, k: int = 50, scoreFunc: str = 'f_classif'):
        """
        Args:
            k: Number of top features to select
            scoreFunc: Scoring function ('f_classif' or 'mutual_info')
        """
        self.k = k
        self.scoreFunc = scoreFunc
        
        if scoreFunc == 'f_classif':
            self.selector = SelectKBest(score_func=f_classif, k=k)
        elif scoreFunc == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            raise ValueError(f"Unknown score function: {scoreFunc}")
            
        self.selectedFeatures = None
        self.scores = None
        
    def fit(self, features: pd.DataFrame, labels: pd.Series) -> 'UnivariateFeatureSelector':
        """Fit selector on features and labels"""
        self.selector.fit(features, labels)
        self.selectedFeatures = features.columns[self.selector.get_support()].tolist()
        self.scores = pd.Series(
            self.selector.scores_,
            index=features.columns
        ).sort_values(ascending=False)
        
        logger.info(f"Univariate selection ({self.scoreFunc}): {len(self.selectedFeatures)} features selected")
        return self
        
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features by selecting top-k features"""
        if self.selectedFeatures is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return features[self.selectedFeatures]
        
    def fitTransform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """Fit and transform features"""
        return self.fit(features, labels).transform(features)
        
    def getFeatureScores(self) -> pd.Series:
        """Get feature importance scores"""
        if self.scores is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self.scores


class TreeBasedFeatureSelector:
    """Select features using tree-based model importance"""
    
    def __init__(
        self,
        modelType: str = 'random_forest',
        nEstimators: int = 100,
        maxFeatures: Optional[int] = None,
        threshold: Union[str, float] = 'median'
    ):
        """
        Args:
            modelType: Type of model ('random_forest' or 'gradient_boosting')
            nEstimators: Number of trees
            maxFeatures: Maximum number of features to select
            threshold: Importance threshold ('mean', 'median', or float value)
        """
        self.modelType = modelType
        self.nEstimators = nEstimators
        self.maxFeatures = maxFeatures
        self.threshold = threshold
        
        if modelType == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=nEstimators,
                random_state=42,
                n_jobs=-1
            )
        elif modelType == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=nEstimators,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {modelType}")
            
        self.selector = None
        self.selectedFeatures = None
        self.featureImportances = None
        
    def fit(self, features: pd.DataFrame, labels: pd.Series) -> 'TreeBasedFeatureSelector':
        """Fit selector on features and labels"""
        # Train model
        self.model.fit(features, labels)
        
        # Get feature importances
        self.featureImportances = pd.Series(
            self.model.feature_importances_,
            index=features.columns
        ).sort_values(ascending=False)
        
        # Create selector
        self.selector = SelectFromModel(
            self.model,
            threshold=self.threshold,
            prefit=True,
            max_features=self.maxFeatures
        )
        
        self.selectedFeatures = features.columns[self.selector.get_support()].tolist()
        
        logger.info(f"Tree-based selection ({self.modelType}): {len(self.selectedFeatures)} features selected")
        return self
        
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features by selecting important features"""
        if self.selectedFeatures is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return features[self.selectedFeatures]
        
    def fitTransform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """Fit and transform features"""
        return self.fit(features, labels).transform(features)
        
    def getFeatureImportances(self) -> pd.Series:
        """Get feature importances"""
        if self.featureImportances is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self.featureImportances


class RecursiveFeatureElimination:
    """Select features using Recursive Feature Elimination (RFE)"""
    
    def __init__(
        self,
        nFeaturesToSelect: int = 50,
        step: int = 1,
        estimator: Optional[object] = None
    ):
        """
        Args:
            nFeaturesToSelect: Number of features to select
            step: Number of features to remove at each iteration
            estimator: Base estimator (default: RandomForestClassifier)
        """
        self.nFeaturesToSelect = nFeaturesToSelect
        self.step = step
        
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                n_jobs=-1
            )
            
        self.selector = RFE(
            estimator=estimator,
            n_features_to_select=nFeaturesToSelect,
            step=step
        )
        
        self.selectedFeatures = None
        self.ranking = None
        
    def fit(self, features: pd.DataFrame, labels: pd.Series) -> 'RecursiveFeatureElimination':
        """Fit RFE on features and labels"""
        self.selector.fit(features, labels)
        self.selectedFeatures = features.columns[self.selector.get_support()].tolist()
        self.ranking = pd.Series(
            self.selector.ranking_,
            index=features.columns
        ).sort_values()
        
        logger.info(f"RFE selection: {len(self.selectedFeatures)} features selected")
        return self
        
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features by selecting RFE-selected features"""
        if self.selectedFeatures is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return features[self.selectedFeatures]
        
    def fitTransform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """Fit and transform features"""
        return self.fit(features, labels).transform(features)
        
    def getFeatureRanking(self) -> pd.Series:
        """Get feature ranking (1 = best)"""
        if self.ranking is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self.ranking


class LassoFeatureSelector:
    """Select features using L1 regularization (Lasso)"""
    
    def __init__(self, alphaRange: Optional[List[float]] = None, cv: int = 5):
        """
        Args:
            alphaRange: Range of alpha values to try
            cv: Number of cross-validation folds
        """
        if alphaRange is None:
            alphaRange = np.logspace(-4, 1, 50)
            
        self.lasso = LassoCV(
            alphas=alphaRange,
            cv=cv,
            random_state=42,
            n_jobs=-1
        )
        
        self.selectedFeatures = None
        self.coefficients = None
        
    def fit(self, features: pd.DataFrame, labels: pd.Series) -> 'LassoFeatureSelector':
        """Fit Lasso on features and labels"""
        self.lasso.fit(features, labels)
        
        # Get non-zero coefficients
        self.coefficients = pd.Series(
            self.lasso.coef_,
            index=features.columns
        )
        
        self.selectedFeatures = self.coefficients[self.coefficients != 0].index.tolist()
        
        logger.info(f"Lasso selection: {len(self.selectedFeatures)} features selected (alpha={self.lasso.alpha_:.4f})")
        return self
        
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features by selecting Lasso-selected features"""
        if self.selectedFeatures is None:
            raise ValueError("Selector not fitted. Call fit() first.")
            
        if len(self.selectedFeatures) == 0:
            logger.warning("No features selected by Lasso. Returning top 10 by coefficient magnitude.")
            self.selectedFeatures = self.coefficients.abs().nlargest(10).index.tolist()
            
        return features[self.selectedFeatures]
        
    def fitTransform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """Fit and transform features"""
        return self.fit(features, labels).transform(features)
        
    def getCoefficients(self) -> pd.Series:
        """Get Lasso coefficients"""
        if self.coefficients is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self.coefficients.abs().sort_values(ascending=False)


class CorrelationFeatureSelector:
    """Remove highly correlated features"""
    
    def __init__(self, threshold: float = 0.95):
        """
        Args:
            threshold: Correlation threshold above which features are removed
        """
        self.threshold = threshold
        self.selectedFeatures = None
        self.correlationMatrix = None
        
    def fit(self, features: pd.DataFrame, labels: Optional[pd.Series] = None) -> 'CorrelationFeatureSelector':
        """Fit selector on features"""
        # Calculate correlation matrix
        self.correlationMatrix = features.corr().abs()
        
        # Find features to remove
        upper = self.correlationMatrix.where(
            np.triu(np.ones(self.correlationMatrix.shape), k=1).astype(bool)
        )
        
        toRemove = [column for column in upper.columns if any(upper[column] > self.threshold)]
        
        self.selectedFeatures = [col for col in features.columns if col not in toRemove]
        
        logger.info(f"Correlation selection: {len(self.selectedFeatures)} features selected "
                   f"({len(toRemove)} removed due to high correlation)")
        return self
        
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features by removing correlated features"""
        if self.selectedFeatures is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return features[self.selectedFeatures]
        
    def fitTransform(self, features: pd.DataFrame, labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform features"""
        return self.fit(features, labels).transform(features)


class EnsembleFeatureSelector:
    """Combine multiple feature selection methods"""
    
    def __init__(
        self,
        methods: Optional[List[str]] = None,
        votingThreshold: float = 0.5,
        maxFeatures: Optional[int] = None
    ):
        """
        Args:
            methods: List of methods to use ('variance', 'univariate', 'tree', 'rfe', 'lasso', 'correlation')
            votingThreshold: Minimum fraction of methods that must select a feature
            maxFeatures: Maximum number of features to select
        """
        if methods is None:
            methods = ['univariate', 'tree', 'correlation']
            
        self.methods = methods
        self.votingThreshold = votingThreshold
        self.maxFeatures = maxFeatures
        
        self.selectors = {}
        self.selectedFeatures = None
        self.featureVotes = None
        
    def fit(self, features: pd.DataFrame, labels: pd.Series) -> 'EnsembleFeatureSelector':
        """Fit all selectors and combine results"""
        print(f"Inside EnsembleFeatureSelector.fit - features.shape: {features.shape}, labels.shape: {labels.shape}", flush=True)
        featureVotes = pd.Series(0, index=features.columns)
        
        # Variance selector
        if 'variance' in self.methods:
            selector = VarianceFeatureSelector()
            selector.fit(features)
            self.selectors['variance'] = selector
            featureVotes[selector.selectedFeatures] += 1
            
        # Univariate selector
        if 'univariate' in self.methods:
            k = min(len(features.columns) // 2, 100)
            selector = UnivariateFeatureSelector(k=k)
            selector.fit(features, labels)
            self.selectors['univariate'] = selector
            featureVotes[selector.selectedFeatures] += 1
            
        # Tree-based selector
        if 'tree' in self.methods:
            selector = TreeBasedFeatureSelector()
            selector.fit(features, labels)
            self.selectors['tree'] = selector
            featureVotes[selector.selectedFeatures] += 1
            
        # RFE selector
        if 'rfe' in self.methods:
            k = min(len(features.columns) // 2, 50)
            selector = RecursiveFeatureElimination(nFeaturesToSelect=k)
            selector.fit(features, labels)
            self.selectors['rfe'] = selector
            featureVotes[selector.selectedFeatures] += 1
            
        # Lasso selector
        if 'lasso' in self.methods:
            selector = LassoFeatureSelector()
            selector.fit(features, labels)
            self.selectors['lasso'] = selector
            featureVotes[selector.selectedFeatures] += 1
            
        # Correlation selector
        if 'correlation' in self.methods:
            selector = CorrelationFeatureSelector()
            selector.fit(features)
            self.selectors['correlation'] = selector
            featureVotes[selector.selectedFeatures] += 1
            
        # Normalize votes
        self.featureVotes = featureVotes / len(self.methods)
        
        # Select features based on voting threshold
        self.selectedFeatures = self.featureVotes[
            self.featureVotes >= self.votingThreshold
        ].index.tolist()
        
        # Limit to maxFeatures if specified
        if self.maxFeatures is not None and len(self.selectedFeatures) > self.maxFeatures:
            topFeatures = self.featureVotes.nlargest(self.maxFeatures).index.tolist()
            self.selectedFeatures = topFeatures
            
        logger.info(f"Ensemble selection: {len(self.selectedFeatures)} features selected "
                   f"(voting threshold: {self.votingThreshold})")
        
        return self
        
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features by selecting ensemble-selected features"""
        if self.selectedFeatures is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return features[self.selectedFeatures]
        
    def fitTransform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """Fit and transform features"""
        return self.fit(features, labels).transform(features)
        
    def getFeatureVotes(self) -> pd.Series:
        """Get feature voting scores"""
        if self.featureVotes is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self.featureVotes.sort_values(ascending=False)


class FeatureSelectionPipeline:
    """Complete feature selection pipeline"""
    
    def __init__(
        self,
        method: str = 'ensemble',
        **kwargs
    ):
        """
        Args:
            method: Selection method ('variance', 'univariate', 'tree', 'rfe', 'lasso', 'correlation', 'ensemble')
            **kwargs: Additional arguments for the selector
        """
        self.method = method
        
        if method == 'variance':
            self.selector = VarianceFeatureSelector(**kwargs)
        elif method == 'univariate':
            self.selector = UnivariateFeatureSelector(**kwargs)
        elif method == 'tree':
            self.selector = TreeBasedFeatureSelector(**kwargs)
        elif method == 'rfe':
            self.selector = RecursiveFeatureElimination(**kwargs)
        elif method == 'lasso':
            self.selector = LassoFeatureSelector(**kwargs)
        elif method == 'correlation':
            self.selector = CorrelationFeatureSelector(**kwargs)
        elif method == 'ensemble':
            self.selector = EnsembleFeatureSelector(**kwargs)
        else:
            raise ValueError(f"Unknown selection method: {method}")
            
    def fit(self, features: pd.DataFrame, labels: Optional[pd.Series] = None) -> 'FeatureSelectionPipeline':
        """Fit selector"""
        self.selector.fit(features, labels)
        return self
        
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features"""
        return self.selector.transform(features)
        
    def fitTransform(self, features: pd.DataFrame, labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform features"""
        return self.selector.fitTransform(features, labels)
        
    def getSelectedFeatures(self) -> List[str]:
        """Get list of selected features"""
        return self.selector.selectedFeatures
