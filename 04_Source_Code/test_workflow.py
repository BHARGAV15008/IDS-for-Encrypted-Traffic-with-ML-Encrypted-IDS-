import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_preprocessing():
    """Test just the preprocessing step"""
    logger.info("Testing preprocessing step...")
    
    # Import the preprocessing pipeline
    try:
        from dataPreprocessor import DataPreprocessingPipeline
        logger.info("Successfully imported DataPreprocessingPipeline")
        
        # Create preprocessor
        preprocessor = DataPreprocessingPipeline(dataType='csv')
        
        # Process data
        data_path = "e:\\IDS for Encrypted Traffic with ML (Encrypted IDS)\\01_Data\\MachineLearningCVE\\Monday-WorkingHours.pcap_ISCX.csv"
        
        logger.info(f"Processing data from: {data_path}")
        features, labels = preprocessor.process(
            inputPath=data_path,
            labelColumn='label',
            handleMissing=True,
            removeOutliers=True,
            encodeCategorical=True,
            normalize=True
        )
        
        logger.info(f"Preprocessing successful! Features shape: {features.shape}, Labels shape: {labels.shape}")
        logger.info(f"Features columns: {len(features.columns)}")
        logger.info(f"Labels unique values: {labels.unique()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_feature_extraction():
    """Test feature extraction after preprocessing"""
    logger.info("Testing feature extraction step...")
    
    try:
        # First test preprocessing
        if not test_preprocessing():
            return False
            
        # Import feature extraction pipelines
        from baseFeatureExtractor import FeatureExtractionPipeline
        from advancedFeatureExtractor import AdvancedFeatureExtractionPipeline
        
        logger.info("Successfully imported feature extraction pipelines")
        
        # Create preprocessor and get data
        preprocessor = DataPreprocessingPipeline(dataType='csv')
        data_path = "e:\\IDS for Encrypted Traffic with ML (Encrypted IDS)\\01_Data\\MachineLearningCVE\\Monday-WorkingHours.pcap_ISCX.csv"
        
        features, labels = preprocessor.process(
            inputPath=data_path,
            labelColumn='label',
            handleMissing=True,
            removeOutliers=True,
            encodeCategorical=True,
            normalize=True
        )
        
        logger.info(f"Original features shape: {features.shape}")
        
        # Extract base features
        baseExtractor = FeatureExtractionPipeline()
        baseFeatures = baseExtractor.extractAll(features)
        logger.info(f"Base features shape: {baseFeatures.shape}")
        
        # Extract advanced features
        advancedExtractor = AdvancedFeatureExtractionPipeline()
        advancedFeatures = advancedExtractor.extractAll(features)
        logger.info(f"Advanced features shape: {advancedFeatures.shape}")
        
        # Combine features
        allFeatures = pd.concat([baseFeatures, advancedFeatures], axis=1)
        allFeatures = allFeatures.loc[:, ~allFeatures.columns.duplicated()]
        logger.info(f"Combined features shape: {allFeatures.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("Starting simplified workflow test...")
    
    # Test just preprocessing first
    success = test_preprocessing()
    
    if success:
        logger.info("Preprocessing test passed!")
        # Now test feature extraction
        success = test_feature_extraction()
        
        if success:
            logger.info("Feature extraction test passed!")
        else:
            logger.error("Feature extraction test failed!")
    else:
        logger.error("Preprocessing test failed!")
        
    logger.info("Test completed.")