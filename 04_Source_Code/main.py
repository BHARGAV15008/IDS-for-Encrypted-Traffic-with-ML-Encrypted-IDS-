#!/usr/bin/env python3
"""
P22 Encrypted Traffic IDS - Main Entry Point

Microservices-based Intrusion Detection System for Encrypted Traffic.

Architecture:
- LSTM Model Service: Temporal pattern detection
- CNN Model Service: Spatial feature extraction
- Data Ingestion Service: CSV and PCAP processing
- Output Management Service: Result aggregation and storage
- Configuration Service: Centralized configuration management

Usage:
    python main.py --help
    python main.py init
    python main.py predict <data_file>
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cli.cliInterface import cli

if __name__ == '__main__':
    cli(obj={})
