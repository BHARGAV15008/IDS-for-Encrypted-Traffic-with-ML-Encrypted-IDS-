# P22 Encrypted Traffic IDS - Microservices Architecture

## 🏗️ Architecture Overview

This implementation follows a **microservices architecture** where each component is independent, modular, and communicates through well-defined interfaces.

### Core Microservices

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface Layer                       │
│              (Terminal-based Interaction)                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Service Orchestrator                       │
│           (Coordinates workflow between services)            │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Data        │     │ LSTM Model  │     │ CNN Model   │
│ Ingestion   │────▶│ Service     │     │ Service     │
│ Service     │     │ (Temporal)  │     │ (Spatial)   │
└─────────────┘     └─────────────┘     └─────────────┘
                            │                   │
                            └────────┬──────────┘
                                     ▼
                           ┌─────────────────┐
                           │ Output          │
                           │ Management      │
                           │ Service         │
                           └─────────────────┘
```

## 📦 Services Description

### 1. **Configuration Service**
- Centralized configuration management
- YAML/JSON configuration support
- Runtime configuration updates
- Configuration validation

**Location:** `services/configurationService.py`

### 2. **Data Ingestion Service**
- Supports **.csv** and **.pcap** file formats
- Automatic feature extraction
- Data normalization and preprocessing
- Metadata extraction

**Location:** `services/dataIngestionService.py`

### 3. **LSTM Model Service**
- Bi-directional LSTM architecture
- Temporal pattern recognition
- Attention mechanism support
- Independent output storage

**Location:** `services/lstmModelService.py`

### 4. **CNN Model Service**
- 1D Convolutional Neural Network
- Multi-scale spatial feature extraction
- Packet-level analysis
- Independent output storage

**Location:** `services/cnnModelService.py`

### 5. **Output Management Service**
- Separated output directories (LSTM/CNN/Ensemble)
- Result aggregation (voting, averaging, max confidence)
- Report generation
- Buffer management

**Location:** `services/outputManagementService.py`

## 🚀 Quick Start

### Installation

```bash
# Navigate to source code directory
cd "04_Source_Code"

# Install dependencies (if not already installed)
pip install -r ../requirements.txt
```

### Basic Usage

#### 1. Export Default Configuration
```bash
python main.py export-config -o my_config.yaml
```

#### 2. Initialize Services
```bash
python main.py init
# or with custom config
python main.py --config my_config.yaml init
```

#### 3. Check Services Status
```bash
python main.py status
```

#### 4. Process Data File
```bash
# Process CSV file
python main.py process data.csv

# Process PCAP file
python main.py process capture.pcap --type pcap
```

#### 5. Run Predictions
```bash
# Run with both models
python main.py predict data.csv

# Run with LSTM only
python main.py predict data.csv --model lstm

# Run with CNN only
python main.py predict data.csv --model cnn

# Run without aggregation
python main.py predict data.csv --no-aggregate
```

#### 6. Generate Report
```bash
python main.py report
```

#### 7. Shutdown Services
```bash
python main.py shutdown
```

## 📁 Output Structure

All outputs are organized in separated directories:

```
outputs/
├── ingestion/                    # Data ingestion outputs
│   ├── *_csv_*_metadata.json
│   ├── *_csv_*_features.npy
│   └── *_pcap_*_features.npy
├── lstm/                         # LSTM model predictions
│   └── lstm_prediction_*.json
├── cnn/                          # CNN model predictions
│   └── cnn_prediction_*.json
├── ensemble/                     # Ensemble predictions
│   └── ensemble_prediction_*.json
└── final/                        # Aggregated final results
    ├── final_prediction_*.json
    └── output_report_*.json
```

## 🔧 Configuration

### Configuration File Format (YAML)

```yaml
system:
  logLevel: INFO
  maxWorkers: 4
  enableGPU: true

dataIngestion:
  csvConfig:
    normalize: true
    labelColumn: label
  pcapConfig:
    maxPackets: 100
    maxPacketSize: 1500

lstmModel:
  inputSize: 256
  hiddenSize: 128
  numLayers: 2
  numClasses: 10
  useAttention: true

cnnModel:
  inputChannels: 1
  sequenceLength: 1500
  numClasses: 10
  convLayers: [64, 128, 256]
  kernelSizes: [3, 5, 7]

outputManagement:
  aggregationMethod: voting  # voting, averaging, or max
```

## 🐍 Programmatic API

### Using the Orchestrator

```python
from orchestrator import ServiceOrchestrator

# Initialize orchestrator
orchestrator = ServiceOrchestrator(configPath='config.yaml')
orchestrator.initialize()

# Run end-to-end pipeline
result = orchestrator.runEndToEndPipeline(
    filePath='data.csv',
    modelType='both',
    aggregate=True
)

# Batch processing
results = orchestrator.batchProcessing(
    fileList=['file1.csv', 'file2.csv'],
    modelType='both'
)

# Get system status
status = orchestrator.getSystemStatus()

# Shutdown
orchestrator.shutdown()
```

### Using Individual Services

```python
from services.lstmModelService import LSTMModelService
from services.cnnModelService import CNNModelService

# Initialize LSTM service
lstmService = LSTMModelService({
    'inputSize': 256,
    'hiddenSize': 128,
    'numClasses': 10
})
lstmService.start()

# Run prediction
result = lstmService.process({
    'features': features_array,
    'metadata': {'source': 'file.csv'}
})

# Get model info
info = lstmService.getModelInfo()

lstmService.stop()
```

## 🎯 Naming Conventions

This project follows **camelCase** naming conventions for:
- Variable names: `inputSize`, `hiddenSize`, `numLayers`
- Method names: `getConfiguration`, `processData`, `runInference`
- Function names: `loadModel`, `saveOutput`, `aggregateResults`

Class names use **PascalCase**: `LSTMModelService`, `CNNModelService`

## 📊 Supported Data Formats

### CSV Files
- Flow-based features
- Statistical features
- Pre-extracted features
- Label column (optional)

**Example CSV:**
```
feature1,feature2,feature3,...,label
0.5,1.2,0.8,...,0
0.3,0.9,1.1,...,1
```

### PCAP Files
- Raw packet captures
- Automatic feature extraction
- Metadata extraction (IPs, ports, protocols)
- Packet-level analysis

## 🔬 Model Architectures

### LSTM Model (Temporal)
- **Architecture:** Bi-directional LSTM
- **Purpose:** Capture temporal dependencies and long-range correlations
- **Features:** 
  - Attention mechanism
  - Variable sequence lengths
  - Dropout regularization

### CNN Model (Spatial)
- **Architecture:** 1D Multi-scale CNN
- **Purpose:** Extract spatial patterns from packet-level data
- **Features:**
  - Multi-scale kernels (3, 5, 7)
  - Batch normalization
  - Max pooling
  - Global average pooling

## 🔄 Aggregation Methods

### 1. Voting (Default)
- Majority voting between models
- Ties resolved by confidence

### 2. Averaging
- Average probability distributions
- Select class with highest average

### 3. Max Confidence
- Select prediction with highest confidence
- Single model decision

## 🧪 Testing

```bash
# Test individual service
python -c "from services.lstmModelService import LSTMModelService; \
           s = LSTMModelService(); s.start(); print(s.getModelInfo()); s.stop()"

# Test orchestrator
python orchestrator.py

# Test CLI
python main.py --help
```

## 📝 Logging

All services provide detailed logging:
- Service lifecycle events (start/stop)
- Processing status
- Errors and warnings
- Performance metrics

Logs are printed to console with timestamps and service names.

## 🔐 Security Considerations

- No hardcoded credentials
- Configuration files should be protected
- Model files should be verified before loading
- Input validation on all data inputs

## 🚧 Extending the System

### Adding a New Service

1. Create service class inheriting from `BaseService`
2. Implement required methods: `_onStart()`, `_onStop()`, `process()`
3. Register in orchestrator
4. Add CLI commands if needed

Example:
```python
from services.baseService import BaseService

class MyNewService(BaseService):
    def __init__(self, config):
        super().__init__("MyNewService", config)
    
    def _onStart(self):
        # Initialization logic
        pass
    
    def _onStop(self):
        # Cleanup logic
        pass
    
    def process(self, data):
        # Processing logic
        return result
```

## 🐛 Troubleshooting

### Services won't start
- Check configuration file syntax
- Verify all dependencies are installed
- Check log output for specific errors

### PCAP processing fails
- Ensure Scapy is installed: `pip install scapy`
- Check file permissions
- Verify PCAP file format

### GPU not detected
- Check CUDA installation
- Verify PyTorch GPU support: `torch.cuda.is_available()`
- Set `enableGPU: false` in config to use CPU

## 📚 References

Based on research from:
- CNN-BiLSTM hybrid architecture
- Adversarial robustness techniques
- Attention mechanisms for IDS
- Ensemble methods for improved detection

See `/References` directory for detailed research papers.

## 🤝 Contributing

Follow the microservices architecture principles:
- Services should be independent
- Use camelCase naming conventions
- Add comprehensive logging
- Document all public methods
- Include type hints

## 📄 License

Part of P22 Encrypted Traffic IDS project.
