"""
Configuration Service

Microservice for centralized configuration management.
Handles loading, validation, and distribution of configuration to all services.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import copy

from .baseService import BaseService


class ConfigurationService(BaseService):
    """Configuration Service for centralized config management."""
    
    # Default configuration schema
    DEFAULT_CONFIG = {
        'system': {
            'logLevel': 'INFO',
            'maxWorkers': 4,
            'enableGPU': True
        },
        'dataIngestion': {
            'csvConfig': {
                'featureColumns': None,
                'labelColumn': 'label',
                'normalize': True
            },
            'pcapConfig': {
                'maxPackets': 100,
                'maxPacketSize': 1500
            },
            'outputDir': './outputs/ingestion'
        },
        'lstmModel': {
            'inputSize': 256,
            'hiddenSize': 128,
            'numLayers': 2,
            'numClasses': 10,
            'dropoutRate': 0.3,
            'useAttention': True,
            'modelPath': None,
            'outputDir': './outputs/lstm'
        },
        'cnnModel': {
            'inputChannels': 1,
            'sequenceLength': 1500,
            'numClasses': 10,
            'convLayers': [64, 128, 256],
            'kernelSizes': [3, 5, 7],
            'dropoutRate': 0.3,
            'modelPath': None,
            'outputDir': './outputs/cnn'
        },
        'outputManagement': {
            'baseOutputDir': './outputs',
            'maxBufferSize': 100,
            'aggregationMethod': 'voting'
        }
    }
    
    def __init__(self, configPath: Optional[str] = None):
        """
        Initialize Configuration Service.
        
        Args:
            configPath: Path to configuration file (YAML or JSON)
        """
        super().__init__("ConfigurationService")
        
        self.configPath = configPath
        self.configuration = copy.deepcopy(self.DEFAULT_CONFIG)
        self.configHistory = []
        
    def _onStart(self) -> None:
        """Load configuration on service start."""
        if self.configPath:
            self.loadConfiguration(self.configPath)
        else:
            self.logger.info("Using default configuration")
        
        # Save initial configuration to history
        self._addToHistory("initialization", self.configuration)
    
    def _onStop(self) -> None:
        """Cleanup on service stop."""
        pass
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process configuration requests.
        
        Args:
            data: Dictionary containing:
                - action: 'get', 'set', 'update', 'reload'
                - key: Configuration key (for get/set)
                - value: Configuration value (for set)
                - updates: Dictionary of updates (for update)
                
        Returns:
            Result dictionary
        """
        if not self.isRunning:
            raise RuntimeError("Configuration Service is not running")
        
        action = data.get('action', 'get')
        
        if action == 'get':
            key = data.get('key')
            return self.getConfiguration(key)
        
        elif action == 'set':
            key = data.get('key')
            value = data.get('value')
            return self.setConfiguration(key, value)
        
        elif action == 'update':
            updates = data.get('updates', {})
            return self.updateConfiguration(updates)
        
        elif action == 'reload':
            path = data.get('path', self.configPath)
            return self.loadConfiguration(path)
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def loadConfiguration(self, configPath: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            configPath: Path to configuration file
            
        Returns:
            Loaded configuration
        """
        try:
            path = Path(configPath)
            
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {configPath}")
            
            # Load based on file extension
            if path.suffix in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    loadedConfig = yaml.safe_load(f)
            elif path.suffix == '.json':
                with open(path, 'r') as f:
                    loadedConfig = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
            
            # Merge with default configuration
            self.configuration = self._mergeConfigs(self.DEFAULT_CONFIG, loadedConfig)
            self.configPath = configPath
            
            # Add to history
            self._addToHistory("loaded", self.configuration, source=configPath)
            
            self.logger.info(f"Loaded configuration from {configPath}")
            
            return {'status': 'success', 'config': self.configuration}
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def saveConfiguration(self, savePath: str, format: str = 'yaml') -> None:
        """
        Save current configuration to file.
        
        Args:
            savePath: Path to save configuration
            format: Output format ('yaml' or 'json')
        """
        try:
            path = Path(savePath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'yaml':
                with open(path, 'w') as f:
                    yaml.dump(self.configuration, f, default_flow_style=False, indent=2)
            elif format == 'json':
                with open(path, 'w') as f:
                    json.dump(self.configuration, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved configuration to {savePath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {str(e)}")
            raise
    
    def getConfiguration(self, key: Optional[str] = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (dot notation supported, e.g., 'lstmModel.hiddenSize')
                 If None, returns entire configuration
                 
        Returns:
            Configuration value
        """
        if key is None:
            return copy.deepcopy(self.configuration)
        
        # Navigate nested configuration using dot notation
        keys = key.split('.')
        value = self.configuration
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(f"Configuration key not found: {key}")
        
        return copy.deepcopy(value)
    
    def setConfiguration(self, key: str, value: Any) -> Dict[str, Any]:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (dot notation supported)
            value: New value
            
        Returns:
            Status dictionary
        """
        # Navigate to parent and set value
        keys = key.split('.')
        config = self.configuration
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        oldValue = config.get(keys[-1])
        config[keys[-1]] = value
        
        # Add to history
        self._addToHistory(
            "set", 
            {key: {'old': oldValue, 'new': value}}
        )
        
        self.logger.info(f"Set configuration: {key} = {value}")
        
        return {
            'status': 'success',
            'key': key,
            'oldValue': oldValue,
            'newValue': value
        }
    
    def updateConfiguration(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            Status dictionary
        """
        updatedKeys = []
        
        for key, value in updates.items():
            try:
                self.setConfiguration(key, value)
                updatedKeys.append(key)
            except Exception as e:
                self.logger.error(f"Failed to update {key}: {str(e)}")
        
        return {
            'status': 'success',
            'updatedKeys': updatedKeys,
            'updateCount': len(updatedKeys)
        }
    
    def validateConfiguration(self) -> Dict[str, Any]:
        """
        Validate current configuration.
        
        Returns:
            Validation result dictionary
        """
        errors = []
        warnings = []
        
        # Validate LSTM configuration
        lstmConfig = self.configuration.get('lstmModel', {})
        if lstmConfig.get('hiddenSize', 0) <= 0:
            errors.append("LSTM hiddenSize must be positive")
        
        # Validate CNN configuration
        cnnConfig = self.configuration.get('cnnModel', {})
        if not cnnConfig.get('convLayers'):
            errors.append("CNN convLayers cannot be empty")
        
        # Validate data ingestion
        dataConfig = self.configuration.get('dataIngestion', {})
        pcapConfig = dataConfig.get('pcapConfig', {})
        if pcapConfig.get('maxPackets', 0) <= 0:
            warnings.append("PCAP maxPackets should be positive")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _mergeConfigs(self, base: Dict, override: Dict) -> Dict:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._mergeConfigs(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _addToHistory(
        self, 
        action: str, 
        data: Any,
        source: Optional[str] = None
    ) -> None:
        """
        Add configuration change to history.
        
        Args:
            action: Action performed
            data: Configuration data
            source: Source of change
        """
        historyEntry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'data': data,
            'source': source
        }
        
        self.configHistory.append(historyEntry)
        
        # Keep only last 100 entries
        if len(self.configHistory) > 100:
            self.configHistory = self.configHistory[-100:]
    
    def getHistory(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get configuration change history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of history entries
        """
        return self.configHistory[-limit:]
    
    def exportDefaultConfig(self, outputPath: str) -> None:
        """
        Export default configuration to file.
        
        Args:
            outputPath: Path to save default configuration
        """
        path = Path(outputPath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(self.DEFAULT_CONFIG, f, default_flow_style=False, indent=2)
        else:
            with open(path, 'w') as f:
                json.dump(self.DEFAULT_CONFIG, f, indent=2)
        
        self.logger.info(f"Exported default configuration to {outputPath}")
