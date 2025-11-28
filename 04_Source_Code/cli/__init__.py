"""
CLI Package for P22 Encrypted Traffic IDS

Terminal-based interface for interacting with the microservices architecture.
"""

__version__ = "1.0.0"

from .cliInterface import IDSCLIManager, cli

__all__ = ['IDSCLIManager', 'cli']
