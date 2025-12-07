"""
Complete Workflow Orchestrator
End-to-end pipeline for encrypted traffic IDS
Integrates all modules: preprocessing, feature engineering, training, evaluation
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Dict, Optional, Tuple
import json
import argparse

# Add project root to path
projectRoot = Path(__file__).parent.parent
sys.path.insert(0, str(projectRoot))
sys.path.insert(0, str(projectRoot / '01_Data' / '02_Processed'))
sys.path.insert(0, str(projectRoot / '02_Features' / '01_Feature_Extraction_Scripts'))
sys.path.insert(0, str(projectRoot / '02_Features' / '02_Novel_Feature_Modules'))
sys.path.insert(0, str(projectRoot / '02_Features' / '03_Feature_Selection_Analysis'))
sys.path.insert(0, str(projectRoot / '03_Models' / '01_Architectures'))
sys.path.insert(0, str(projectRoot / '03_Models' / '02_Training_Scripts'))
sys.path.insert(0, str(projectRoot / '03_Models' / '04_Adversarial_Training'))
sys.path.insert(0, str(projectRoot / '05_Evaluation' / '01_Metrics_Calculators'))
sys.path.insert(0, str(projectRoot / '05_Evaluation' / '04_Visualization_Scripts'))

# Import custom modules
from dataPreprocessor import DataPreprocessingPipeline
from baseFeatureExtractor import FeatureExtractionPipeline
from advancedFeatureExtractor import AdvancedFeatureExtractionPipeline
from featureEngineer import FeatureEngineeringPipeline
from featureSelector import FeatureSelectionPipeline
from hybridCnnBiLstmAttention import HybridCnnBiLstmAttention, HybridModelFactory
from ensembleModule import HybridDeepEnsemble
from modelTrainer import ModelTrainer
from hyperparameterTuner import OptunaTuner
from adversarialTrainer import AdversarialTrainer, RobustnessEvaluator
from metricsCalculator import MetricsCalculator, AdversarialMetricsCalculator
from changeDetectionModule import EnsembleChangeDetector
from performanceVisualizer import PerformanceVisualizer

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


class CompleteIDSWorkflow:
    """
    Complete end-to-end workflow for encrypted traffic IDS
    Handles everything from data loading to deployment
    """
    
    def __init__(
        self,
        dataPath: str,
        dataType: str = 'csv',
        outputDir: str = 'outputs',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            dataPath: Path to input data (CSV or PCAP)
            dataType: Type of data ('csv' or 'pcap')
            outputDir: Directory for outputs
            device: Device to use
        """
        self.dataPath = dataPath
        self.dataType = dataType
        self.outputDir = Path(outputDir)
        self.device = device
        
        self.outputDir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.preprocessor = None
        self.featureExtractor = None
        self.featureEngineer = None
        self.featureSelector = None
        self.model = None
        self.trainer = None
        self.changeDetector = None
        self.visualizer = PerformanceVisualizer(outputDir=str(self.outputDir / 'visualizations'))
        
        # Store results
        self.results = {
            'preprocessing': {},
            'features': {},
            'training': {},
            'evaluation': {},
            'robustness': {}
        }
        
        logger.info(f"Workflow initialized: data_type={dataType}, device={device}")
        
    def runComplete(
        self,
        labelColumn: str = 'label',
        numEpochs: int = 100,
        batchSize: int = 64,
        learningRate: float = 0.001,
        useHyperparameterTuning: bool = False,
        useAdversarialTraining: bool = True,
        saveModel: bool = True
    ) -> Dict:
        """
        Run complete workflow
        
        Args:
            labelColumn: Name of label column
            numEpochs: Number of training epochs
            batchSize: Batch size
            learningRate: Learning rate
            useHyperparameterTuning: Whether to tune hyperparameters
            useAdversarialTraining: Whether to use adversarial training
            saveModel: Whether to save trained model
            
        Returns:
            Dictionary of results
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING COMPLETE IDS WORKFLOW")
            
            # Step 1: Data Preprocessing
            logger.info("\n[STEP 1/7] Data Preprocessing")
            features, labels = self.preprocessData(labelColumn)
            if not all(col in features.columns for col in ['srcIP', 'srcPort', 'dstIP', 'dstPort', 'protocol']):
                logger.warning("The preprocessed data is missing one or more of the required columns ('srcIP', 'srcPort', 'dstIP', 'dstPort', 'protocol') to create a flowId.")
                logger.warning("This will likely cause issues with feature extraction.")
            with open(r"e:\IDS for Encrypted Traffic with ML (Encrypted IDS)\04_Source_Code\debug_shapes.log", "a") as f:
                f.write(f"After preprocessData - features.shape: {features.shape}, labels.shape: {labels.shape}\n")
                if 'flowId' in features.columns:
                    f.write(f"Number of unique flowIds: {features['flowId'].nunique()}\n")
            logger.info(f"After preprocessing - Features shape: {features.shape}, Labels shape: {labels.shape}")
            
            # Step 2: Feature extraction (skip for time series)
            if self.dataType in ['timeseries', 'npz']:
                logger.info("\n[STEP 2/7] Feature Extraction - SKIPPED (time series data)")
                logger.info("Time series data already contains engineered features")
                extractedFeatures = features
            else:
                logger.info("\n[STEP 2/7] Feature Extraction")
                extractedFeatures = self.extractFeatures(features)
                with open(r"e:\IDS for Encrypted Traffic with ML (Encrypted IDS)\04_Source_Code\debug_shapes.log", "a") as f:
                    f.write(f"After extractFeatures - features.shape: {extractedFeatures.shape}\n")
                logger.info(f"After feature extraction - Features shape: {extractedFeatures.shape}")
            
            # Step 3: Feature engineering (skip for time series)
            if self.dataType in ['timeseries', 'npz']:
                logger.info("\n[STEP 3/7] Feature Engineering - SKIPPED (time series data)")
                engineeredFeatures = extractedFeatures
            else:
                logger.info("\n[STEP 3/7] Feature Engineering")
                engineeredFeatures = self.engineerFeatures(extractedFeatures)
                with open(r"e:\IDS for Encrypted Traffic with ML (Encrypted IDS)\04_Source_Code\debug_shapes.log", "a") as f:
                    f.write(f"After engineerFeatures - features.shape: {engineeredFeatures.shape}\n")
                logger.info(f"After feature engineering - Features shape: {engineeredFeatures.shape}")
            
            
            print(f"Shape of engineeredFeatures before reset: {engineeredFeatures.shape}")
            print(f"Shape of labels before reset: {labels.shape}")

            # Reset index to ensure alignment
            engineeredFeatures.reset_index(drop=True, inplace=True)
            labels.reset_index(drop=True, inplace=True)

            print(f"Shape of engineeredFeatures after reset: {engineeredFeatures.shape}")
            print(f"Shape of labels after reset: {labels.shape}")

            # Step 4: Feature Selection
            logger.info("\n[STEP 4/7] Feature Selection")
            logger.info(f"Before feature selection - Features shape: {engineeredFeatures.shape}, Labels shape: {labels.shape}")
            selectedFeatures = self.selectFeatures(engineeredFeatures, labels)
            
            # Step 5: Model Training
            logger.info("\n[STEP 5/7] Model Training")
            if useHyperparameterTuning:
                self.tuneHyperparameters(selectedFeatures, labels, batchSize)
            
            loss_config = {
                'name': 'focal_loss',
                'gamma': 2.0,
                'alpha': 'auto'
            }
            
            trainedModel = self.trainModel(
                selectedFeatures, labels, numEpochs, batchSize, learningRate,
                useAdversarialTraining, loss_config=loss_config
            )
            
            # Step 6: Model Evaluation
            logger.info("\n[STEP 6/7] Model Evaluation")
            evalMetrics = self.evaluateModel(selectedFeatures, labels)
            
            # Step 7: Change Detection Setup
            logger.info("\n[STEP 7/7] Change Detection Setup")
            self.setupChangeDetection(selectedFeatures)
            
            # Step 8: Generate Visualizations
            logger.info("\n[STEP 8/8] Generating Visualizations")
            self.generateVisualizations(selectedFeatures, labels)
            
            # Save results
            if saveModel:
                self.saveResults()
                
            logger.info("\n" + "=" * 80)
            logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return self.results
        except Exception as e:
            with open('error.log', 'w') as f:
                f.write(f"An error occurred: {e}")
            print(f"An error occurred: {e}")
            raise
        
    def preprocessData(self, labelColumn: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess data"""
        logger.info(f"Loading data from: {self.dataPath}")
        
        # Handle time series data (NPZ format)
        if self.dataType in ['timeseries', 'npz']:
            logger.info("Loading time series data from NPZ file...")
            data = np.load(self.dataPath)
            
            sequences = data['sequences']  # Shape: (N, window_size, features)
            labels = data['labels'] if 'labels' in data else None
            
            logger.info(f"✓ Loaded {sequences.shape[0]} sequences")
            logger.info(f"  - Window size: {sequences.shape[1]}")
            logger.info(f"  - Features: {sequences.shape[2]}")
            
            # For time series, we'll reshape to 2D for compatibility
            # Shape: (N, window_size * features)
            N, window_size, num_features = sequences.shape
            features_flat = sequences.reshape(N, window_size * num_features)
            
            # Convert to DataFrame
            feature_names = [f'ts_{i}_{j}' for i in range(window_size) for j in range(num_features)]
            features = pd.DataFrame(features_flat, columns=feature_names)
            labels = pd.Series(labels) if labels is not None else pd.Series([0] * len(features))
            
            self.results['preprocessing'] = {
                'num_samples': len(features),
                'num_features': len(features.columns),
                'num_classes': len(labels.unique()),
                'class_distribution': labels.value_counts().to_dict(),
                'is_timeseries': True,
                'window_size': window_size,
                'original_features': num_features
            }
            
            logger.info(f"✓ Preprocessed {len(features)} samples with {len(features.columns)} features")
            
        else:
            # Standard CSV/PCAP preprocessing
            self.preprocessor = DataPreprocessingPipeline(dataType=self.dataType)
            
            features, labels = self.preprocessor.process(
                inputPath=self.dataPath,
                labelColumn=labelColumn,
                handleMissing=True,
                removeOutliers=False,
                encodeCategorical=True,
                normalize=True
            )
            
            self.results['preprocessing'] = {
                'num_samples': len(features),
                'num_features': len(features.columns),
                'num_classes': len(labels.unique()),
                'class_distribution': labels.value_counts().to_dict()
            }
            
            logger.info(f"✓ Preprocessed {len(features)} samples with {len(features.columns)} features")

            # Create a flowId for each row if it doesn't exist
            if 'flowId' not in features.columns:
                features['flowId'] = features.index
                logger.info("Created 'flowId' column from DataFrame index.")
        
        features.set_index('flowId', inplace=True)
        labels.index = features.index
        
        return features, labels
        
    def extractFeatures(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features"""
        logger.info("Extracting base and advanced features...")
        
        # Base features
        baseExtractor = FeatureExtractionPipeline()
        baseFeatures = baseExtractor.extractAll(data)
        
        # Advanced features
        advancedExtractor = AdvancedFeatureExtractionPipeline()
        advancedFeatures = advancedExtractor.extractAll(data)
        
        # Combine
        allFeatures = pd.concat([baseFeatures, advancedFeatures], axis=1)
        allFeatures = allFeatures.loc[:, ~allFeatures.columns.duplicated()]
        
        self.results['features']['extracted'] = len(allFeatures.columns)
        
        logger.info(f"✓ Extracted {len(allFeatures.columns)} features")
        
        return allFeatures
        
    def engineerFeatures(self, features: pd.DataFrame) -> pd.DataFrame:
        """Engineer features"""
        logger.info("Engineering features...")
        logger.info(f"Shape before feature engineering: {features.shape}")
        
        self.featureEngineer = FeatureEngineeringPipeline(
            transformMethod='robust',
            createInteractions=True,
            createInvariant=True,
            createDomainSpecific=True
        )
        
        engineeredFeatures = self.featureEngineer.fitTransform(features)
        logger.info(f"Shape after feature engineering: {engineeredFeatures.shape}")
        
        self.results['features']['engineered'] = len(engineeredFeatures.columns)
        
        logger.info(f"✓ Engineered {len(engineeredFeatures.columns)} features")
        
        return engineeredFeatures
        
    def selectFeatures(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """Selects the best features from the dataset."""
        with open("debug_shapes.log", "a") as f:
            f.write(f"Inside selectFeatures - initial features.shape: {features.shape}, labels.shape: {labels.shape}\n")
        logger.info("Selecting best features...")
        
        self.featureSelector = FeatureSelectionPipeline(
            method='ensemble',
            votingThreshold=0.5,
            maxFeatures=100
        )
        
        with open("debug_shapes.log", "a") as f:
            f.write(f"Inside selectFeatures - before fit_transform - features.shape: {features.shape}, labels.shape: {labels.shape}\n")
        selectedFeatures = self.featureSelector.fitTransform(features, labels)
        
        self.results['features']['selected'] = len(selectedFeatures.columns)
        self.results['features']['feature_names'] = selectedFeatures.columns.tolist()
        
        logger.info(f"✓ Selected {len(selectedFeatures.columns)} features")
        
        return selectedFeatures
        
    def tuneHyperparameters(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        batchSize: int
    ):
        """Tune hyperparameters"""
        logger.info("Tuning hyperparameters with Optuna...")
        
        # Split data
        xTrain, xVal, xTest, yTrain, yVal, yTest = self.preprocessor.splitter.split(
            features, labels
        )
        
        # Create data loaders
        trainDataset = TensorDataset(
            torch.FloatTensor(xTrain.values),
            torch.LongTensor(yTrain.values)
        )
        valDataset = TensorDataset(
            torch.FloatTensor(xVal.values),
            torch.LongTensor(yVal.values)
        )
        
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
        valLoader = DataLoader(valDataset, batch_size=batchSize)
        
        # Initialize model
        model_config = {
            'modelType': 'deep',
            'inputFeatures': features.shape[1],
            'numClasses': len(labels.unique()),
            'cnnChannels': [64, 128, 256, 512],
            'lstmHiddenSize': 256,
            'numAttentionLayers': 3,
            'dropout': 0.3
        }
        
        self.model = HybridModelFactory.create(**model_config)
        self.results['model_architecture'] = model_config
        
        # Log model architecture
        logger.info(f"Model created: {self.model}")
        
        # Train
        self.trainer = ModelTrainer(
            self.model,
            device=self.device,
            checkpointDir=str(self.outputDir / 'training')
        )
        
        # Train the model
        history = self.trainer.train(
            trainLoader,
            valLoader,
            numEpochs=numEpochs,
            learningRate=learningRate,
            loss_config=loss_config,
            useEarlyStopping=True,
            useLrScheduler=True
        )

        self.results['training']['history'] = history

        logger.info("✓ Model training completed")

        return self.model
        
    def evaluateModel(self, features: pd.DataFrame, labels: pd.Series) -> Dict:
        """Evaluate model"""
        logger.info("Evaluating model...")
        
        # Create test loader
        testDataset = TensorDataset(self.xTest, self.yTest)
        testLoader = DataLoader(testDataset, batch_size=64)
        
        # Standard evaluation
        metrics = self.trainer.evaluate(testLoader)
        
        # Calculate comprehensive metrics
        metricsCalc = MetricsCalculator(
            numClasses=len(labels.unique()),
            classNames=[f'Class_{i}' for i in range(len(labels.unique()))]
        )
        
        yPred = np.array(metrics['predictions'])
        yTrue = np.array(metrics['targets'])
        
        allMetrics = metricsCalc.calculateAll(yTrue, yPred)
        
        self.results['evaluation']['metrics'] = allMetrics
        self.results['evaluation']['test_accuracy'] = metrics['test_acc']
        self.results['evaluation']['test_loss'] = metrics['test_loss']
        
        logger.info(f"✓ Test Accuracy: {metrics['test_acc']:.4f}")
        logger.info(f"✓ Detection Rate: {allMetrics.get('detection_rate', 0):.4f}")
        logger.info(f"✓ False Alarm Rate: {allMetrics.get('false_alarm_rate', 0):.4f}")
        
        # Adversarial robustness evaluation
        logger.info("Evaluating adversarial robustness...")
        
        robustnessEval = RobustnessEvaluator(self.model)
        epsilons = [0.001, 0.005, 0.01, 0.05, 0.1]
        robustnessResults = robustnessEval.evaluateRobustness(
            self.xTest, self.yTest,
            epsilons=epsilons
        )
        
        # Store for visualization
        self.results['robustness'] = robustnessResults
        self.results['robustness']['epsilons'] = epsilons
        self.results['robustness']['accuracies'] = [
            robustnessResults.get(f'acc_eps_{eps}', 0) for eps in epsilons
        ]
        
        logger.info(f"✓ Robustness evaluation completed")
        
        return allMetrics
        
    def setupChangeDetection(self, features: pd.DataFrame):
        """Setup change detection"""
        logger.info("Setting up change detection...")
        
        self.changeDetector = EnsembleChangeDetector(
            inputDim=features.shape[1],
            device=self.device
        )
        
        # Train novelty detector on training data
        xTrain = torch.FloatTensor(features.values[:int(len(features) * 0.7)])
        self.changeDetector.trainNoveltyDetector(xTrain.to(self.device))
        
        logger.info("✓ Change detection setup completed")
        
    def generateVisualizations(self, features: pd.DataFrame, labels: pd.Series):
        """Generate all visualizations"""
        logger.info("Generating comprehensive visualizations...")
        
        # Get predictions and probabilities
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.xTest)
            yProba = torch.softmax(outputs, dim=1).cpu().numpy()
            yPred = outputs.argmax(dim=1).cpu().numpy()
            
        yTrue = self.yTest.cpu().numpy()
        
        # 1. Training curves
        if 'history' in self.results['training']:
            logger.info("  - Creating training curves...")
            self.visualizer.plotTrainingCurves(self.results['training']['history'])
            
        # 2. Confusion matrix
        logger.info("  - Creating confusion matrix...")
        classNames = [f'Class_{i}' for i in range(len(labels.unique()))]
        self.visualizer.plotConfusionMatrix(yTrue, yPred, classNames, normalize=True)
        
        # 3. ROC curve
        logger.info("  - Creating ROC curve...")
        self.visualizer.plotROCCurve(yTrue, yProba, classNames)
        
        # 4. Precision-Recall curve
        logger.info("  - Creating Precision-Recall curve...")
        self.visualizer.plotPrecisionRecallCurve(yTrue, yProba, classNames)
        
        # 5. Metrics comparison
        logger.info("  - Creating metrics comparison...")
        self.visualizer.plotMetricsComparison(self.results['evaluation']['metrics'])
        
        # 6. Class distribution
        logger.info("  - Creating class distribution...")
        xTrain, xVal, xTest, yTrain, yVal, yTest = self.preprocessor.splitter.split(
            features, labels
        )
        self.visualizer.plotClassDistribution(
            yTrain.values, yTest.values, classNames
        )
        
        # 7. Adversarial robustness
        if 'epsilons' in self.results['robustness'] and 'accuracies' in self.results['robustness']:
            logger.info("  - Creating adversarial robustness plot...")
            self.visualizer.plotAdversarialRobustness(
                self.results['robustness']['epsilons'],
                self.results['robustness']['accuracies']
            )
            
        # 8. Comprehensive dashboard
        logger.info("  - Creating comprehensive dashboard...")
        if 'history' in self.results['training']:
            self.visualizer.createComprehensiveDashboard(
                history=self.results['training']['history'],
                yTrue=yTrue,
                yPred=yPred,
                yProba=yProba,
                classNames=classNames
            )
            
        logger.info(f"✓ All visualizations saved to: {self.visualizer.outputDir}")
        
    def saveResults(self):
        """Save results and model"""
        logger.info("Saving results...")
        
        # Save model
        modelPath = self.outputDir / 'trained_model.pth'
        torch.save(self.model.state_dict(), modelPath)
        logger.info(f"✓ Model saved to: {modelPath}")
        
        # Save results
        resultsPath = self.outputDir / 'results.json'
        with open(resultsPath, 'w') as f:
            # Convert numpy types to native Python types
            resultsJson = json.dumps(self.results, indent=4, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else str(x))
            f.write(resultsJson)
        logger.info(f"✓ Results saved to: {resultsPath}")
        
        # Save feature names
        if 'feature_names' in self.results['features']:
            featuresPath = self.outputDir / 'selected_features.txt'
            with open(featuresPath, 'w') as f:
                for feat in self.results['features']['feature_names']:
                    f.write(f"{feat}\n")
            logger.info(f"✓ Feature names saved to: {featuresPath}")
            
    def predict(self, newData: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Predict on new data with change detection
        
        Args:
            newData: New data to predict
            
        Returns:
            predictions: Predicted labels
            changeResults: Change detection results
        """
        # Preprocess
        if self.featureEngineer is not None:
            newData = self.featureEngineer.transform(newData)
        if self.featureSelector is not None:
            newData = newData[self.featureSelector.getSelectedFeatures()]
            
        # Convert to tensor
        xNew = torch.FloatTensor(newData.values).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(xNew)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            
        # Change detection
        changeResults = None
        if self.changeDetector is not None:
            changeResults = self.changeDetector.detect(xNew)
            
        return predictions, changeResults


import yaml

def main():
    """Main function for CLI usage"""
    # Load config to get default dataPath
    with open("traffic_classification_configuration.yaml", 'r') as f:
        config = yaml.safe_load(f)
    default_data_path = config['data']['path']

    parser = argparse.ArgumentParser(description="Complete IDS Workflow")
    parser.add_argument("--dataPath", type=str, default=default_data_path, help="Path to input data")
    parser.add_argument("--dataType", type=str, default='csv', help="Type of data (csv or pcap)")
    parser.add_argument("--outputDir", type=str, default="e:\\IDS for Encrypted Traffic with ML (Encrypted IDS)\\03_Models\\03_Trained_Weights", help="Directory for outputs")
    parser.add_argument("--labelColumn", type=str, default='label', help="Name of label column")
    parser.add_argument("--numEpochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batchSize", type=int, default=64, help="Batch size")
    parser.add_argument("--learningRate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--tune", action='store_true', help="Tune hyperparameters")
    parser.add_argument("--adversarial", action='store_true', help="Use adversarial training")
    parser.add_argument("--save", action='store_true', help="Save trained model")
    args = parser.parse_args()

    # Initialize workflow
    workflow = CompleteIDSWorkflow(
        dataPath=args.dataPath,
        dataType=args.dataType,
        outputDir=args.outputDir
    )
    
    # Run workflow
    workflow.runComplete(
        labelColumn=args.labelColumn,
        numEpochs=args.numEpochs,
        batchSize=args.batchSize,
        learningRate=args.learningRate,
        useHyperparameterTuning=args.tune,
        useAdversarialTraining=args.adversarial,
        saveModel=args.save
    )

if __name__ == '__main__':
    main()
