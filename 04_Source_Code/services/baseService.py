"""
Base Service Class for Microservices Architecture

Provides common functionality for all services including:
- Lifecycle management (start/stop)
- Health checks
- Logging
- Configuration management
"""

import logging
import abc
from typing import Dict, Any, Optional
from datetime import datetime
import threading


class BaseService(abc.ABC):
    """Abstract base class for all microservices."""
    
    def __init__(self, serviceName: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base service.
        
        Args:
            serviceName: Unique service identifier
            config: Service configuration dictionary
        """
        self.serviceName = serviceName
        self.config = config or {}
        self.isRunning = False
        self.startTime = None
        self.logger = self._setupLogger()
        self._lock = threading.Lock()
        
    def _setupLogger(self) -> logging.Logger:
        """Setup service-specific logger."""
        logger = logging.getLogger(f"P22.{self.serviceName}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.serviceName} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start(self) -> bool:
        """
        Start the service.
        
        Returns:
            True if service started successfully
        """
        with self._lock:
            if self.isRunning:
                self.logger.warning(f"{self.serviceName} is already running")
                return False
            
            try:
                self.logger.info(f"Starting {self.serviceName}...")
                self._onStart()
                self.isRunning = True
                self.startTime = datetime.now()
                self.logger.info(f"{self.serviceName} started successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to start {self.serviceName}: {str(e)}")
                return False
    
    def stop(self) -> bool:
        """
        Stop the service.
        
        Returns:
            True if service stopped successfully
        """
        with self._lock:
            if not self.isRunning:
                self.logger.warning(f"{self.serviceName} is not running")
                return False
            
            try:
                self.logger.info(f"Stopping {self.serviceName}...")
                self._onStop()
                self.isRunning = False
                self.logger.info(f"{self.serviceName} stopped successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to stop {self.serviceName}: {str(e)}")
                return False
    
    def healthCheck(self) -> Dict[str, Any]:
        """
        Perform health check on the service.
        
        Returns:
            Dictionary containing health status
        """
        uptime = None
        if self.startTime:
            uptime = (datetime.now() - self.startTime).total_seconds()
        
        return {
            'serviceName': self.serviceName,
            'status': 'running' if self.isRunning else 'stopped',
            'uptime': uptime,
            'timestamp': datetime.now().isoformat()
        }
    
    def getConfig(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def updateConfig(self, updates: Dict[str, Any]) -> None:
        """
        Update service configuration.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self.config.update(updates)
        self.logger.info(f"Configuration updated for {self.serviceName}")
        self._onConfigUpdate(updates)
    
    @abc.abstractmethod
    def _onStart(self) -> None:
        """Service-specific startup logic."""
        pass
    
    @abc.abstractmethod
    def _onStop(self) -> None:
        """Service-specific shutdown logic."""
        pass
    
    def _onConfigUpdate(self, updates: Dict[str, Any]) -> None:
        """
        Handle configuration updates.
        
        Args:
            updates: Dictionary of configuration updates
        """
        pass
    
    @abc.abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process data through the service.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed output
        """
        pass
