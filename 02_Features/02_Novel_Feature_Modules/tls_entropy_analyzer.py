"""
TLS Handshake Entropy Analysis - Novel Feature Module for P22 IDS

This module implements innovative entropy-based features for encrypted traffic analysis,
focusing on TLS handshake patterns and encrypted payload characteristics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from collections import Counter
import logging
from dataclasses import dataclass


@dataclass
class TLSHandshakeFeatures:
    """Data class for TLS handshake-derived features."""
    
    # Entropy measures
    cipher_suite_entropy: float
    certificate_entropy: float
    extension_entropy: float
    
    # Timing patterns
    handshake_duration: float
    round_trip_variance: float
    
    # Size characteristics
    certificate_chain_length: int
    total_handshake_bytes: int
    
    # Protocol specifics
    tls_version: str
    cipher_strength: int
    perfect_forward_secrecy: bool


class TLSEntropyAnalyzer:
    """
    Advanced TLS entropy analysis for encrypted traffic feature extraction.
    
    This analyzer implements novel invariance-driven features that remain
    consistent across different network conditions and attack scenarios.
    """
    
    def __init__(self, min_entropy_threshold: float = 0.5):
        """
        Initialize TLS entropy analyzer.
        
        Args:
            min_entropy_threshold: Minimum entropy threshold for valid analysis
        """
        self.min_entropy_threshold = min_entropy_threshold
        self.logger = logging.getLogger(__name__)
        
        # TLS cipher suite mappings
        self.cipher_strength_map = {
            'AES_256': 256,
            'AES_128': 128,
            'ChaCha20': 256,
            'RC4': 128,  # Weak
            '3DES': 112,  # Weak
        }
        
        # Known weak cipher patterns
        self.weak_ciphers = {
            'RC4', 'DES', '3DES', 'MD5', 'SHA1_ONLY'
        }
    
    def extract_handshake_features(self, tls_flows: List[Dict]) -> pd.DataFrame:
        """
        Extract comprehensive TLS handshake features from flow data.
        
        Args:
            tls_flows: List of TLS flow dictionaries
            
        Returns:
            DataFrame with extracted TLS features
        """
        features = []
        
        for flow in tls_flows:
            try:
                flow_features = self._analyze_single_handshake(flow)
                features.append(flow_features)
            except Exception as e:
                self.logger.warning(f"Failed to analyze flow: {e}")
                # Add default features for failed analysis
                features.append(self._get_default_features())
        
        return pd.DataFrame([vars(f) for f in features])
    
    def _analyze_single_handshake(self, flow: Dict) -> TLSHandshakeFeatures:
        """Analyze a single TLS handshake flow."""
        
        # Extract basic handshake information
        handshake_packets = flow.get('handshake_packets', [])
        certificates = flow.get('certificates', [])
        extensions = flow.get('extensions', [])
        
        # Calculate entropy measures
        cipher_entropy = self._calculate_cipher_suite_entropy(flow.get('cipher_suites', []))
        cert_entropy = self._calculate_certificate_entropy(certificates)
        ext_entropy = self._calculate_extension_entropy(extensions)
        
        # Timing analysis
        handshake_duration = self._calculate_handshake_duration(handshake_packets)
        rtt_variance = self._calculate_rtt_variance(handshake_packets)
        
        # Size characteristics
        cert_chain_length = len(certificates)
        total_bytes = sum(pkt.get('size', 0) for pkt in handshake_packets)
        
        # Protocol analysis
        tls_version = flow.get('tls_version', 'unknown')
        cipher_strength = self._assess_cipher_strength(flow.get('cipher_suite', ''))
        pfs = self._check_perfect_forward_secrecy(flow.get('key_exchange', ''))
        
        return TLSHandshakeFeatures(
            cipher_suite_entropy=cipher_entropy,
            certificate_entropy=cert_entropy,
            extension_entropy=ext_entropy,
            handshake_duration=handshake_duration,
            round_trip_variance=rtt_variance,
            certificate_chain_length=cert_chain_length,
            total_handshake_bytes=total_bytes,
            tls_version=tls_version,
            cipher_strength=cipher_strength,
            perfect_forward_secrecy=pfs
        )
    
    def _calculate_cipher_suite_entropy(self, cipher_suites: List[str]) -> float:
        """Calculate entropy of cipher suite selection patterns."""
        if not cipher_suites:
            return 0.0
        
        # Count cipher suite frequencies
        cipher_counts = Counter(cipher_suites)
        total_count = len(cipher_suites)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in cipher_counts.values():
            probability = count / total_count
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_certificate_entropy(self, certificates: List[Dict]) -> float:
        """Calculate entropy based on certificate characteristics."""
        if not certificates:
            return 0.0
        
        # Extract certificate features for entropy calculation
        cert_features = []
        for cert in certificates:
            # Certificate size entropy
            size = cert.get('size', 0)
            # Issuer entropy (based on issuer diversity)
            issuer = cert.get('issuer', '')
            # Validity period entropy
            validity_days = cert.get('validity_days', 0)
            
            cert_features.extend([size, len(issuer), validity_days])
        
        if not cert_features:
            return 0.0
        
        # Normalize features and calculate entropy
        normalized_features = self._normalize_features(cert_features)
        return self._calculate_distribution_entropy(normalized_features)
    
    def _calculate_extension_entropy(self, extensions: List[Dict]) -> float:
        """Calculate entropy from TLS extensions."""
        if not extensions:
            return 0.0
        
        # Count extension types
        ext_types = [ext.get('type', '') for ext in extensions]
        ext_counts = Counter(ext_types)
        
        # Calculate entropy of extension distribution
        total_count = len(ext_types)
        entropy = 0.0
        
        for count in ext_counts.values():
            probability = count / total_count
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_handshake_duration(self, handshake_packets: List[Dict]) -> float:
        """Calculate total handshake duration."""
        if len(handshake_packets) < 2:
            return 0.0
        
        timestamps = [pkt.get('timestamp', 0) for pkt in handshake_packets]
        timestamps.sort()
        
        return timestamps[-1] - timestamps[0]
    
    def _calculate_rtt_variance(self, handshake_packets: List[Dict]) -> float:
        """Calculate round-trip time variance during handshake."""
        if len(handshake_packets) < 4:  # Need at least 2 round trips
            return 0.0
        
        # Extract client-server packet pairs
        client_packets = [pkt for pkt in handshake_packets if pkt.get('direction') == 'client_to_server']
        server_packets = [pkt for pkt in handshake_packets if pkt.get('direction') == 'server_to_client']
        
        if len(client_packets) < 2 or len(server_packets) < 2:
            return 0.0
        
        # Calculate RTTs
        rtts = []
        for i in range(min(len(client_packets), len(server_packets))):
            client_time = client_packets[i].get('timestamp', 0)
            server_time = server_packets[i].get('timestamp', 0)
            if server_time > client_time:
                rtts.append(server_time - client_time)
        
        return np.var(rtts) if len(rtts) > 1 else 0.0
    
    def _assess_cipher_strength(self, cipher_suite: str) -> int:
        """Assess the cryptographic strength of the cipher suite."""
        if not cipher_suite:
            return 0
        
        # Check for known cipher strengths
        for cipher, strength in self.cipher_strength_map.items():
            if cipher in cipher_suite.upper():
                return strength
        
        # Check for weak ciphers
        for weak_cipher in self.weak_ciphers:
            if weak_cipher in cipher_suite.upper():
                return 64  # Weak cipher indicator
        
        return 128  # Default moderate strength
    
    def _check_perfect_forward_secrecy(self, key_exchange: str) -> bool:
        """Check if the connection uses Perfect Forward Secrecy."""
        pfs_indicators = ['DHE', 'ECDHE', 'ECDH_anon']
        return any(indicator in key_exchange.upper() for indicator in pfs_indicators)
    
    def _normalize_features(self, features: List[float]) -> np.ndarray:
        """Normalize feature values for entropy calculation."""
        features_array = np.array(features)
        if np.std(features_array) == 0:
            return features_array
        
        return (features_array - np.mean(features_array)) / np.std(features_array)
    
    def _calculate_distribution_entropy(self, values: np.ndarray, bins: int = 10) -> float:
        """Calculate entropy of a continuous distribution."""
        if len(values) == 0:
            return 0.0
        
        # Create histogram
        hist, _ = np.histogram(values, bins=bins, density=True)
        
        # Calculate entropy
        entropy = 0.0
        for count in hist:
            if count > 0:
                entropy -= count * np.log2(count + 1e-10)  # Add small epsilon to avoid log(0)
        
        return entropy
    
    def _get_default_features(self) -> TLSHandshakeFeatures:
        """Return default features for failed analysis."""
        return TLSHandshakeFeatures(
            cipher_suite_entropy=0.0,
            certificate_entropy=0.0,
            extension_entropy=0.0,
            handshake_duration=0.0,
            round_trip_variance=0.0,
            certificate_chain_length=0,
            total_handshake_bytes=0,
            tls_version='unknown',
            cipher_strength=0,
            perfect_forward_secrecy=False
        )
    
    def calculate_anomaly_scores(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate anomaly scores based on TLS entropy features.
        
        Args:
            features_df: DataFrame with TLS features
            
        Returns:
            Array of anomaly scores (higher = more anomalous)
        """
        anomaly_scores = np.zeros(len(features_df))
        
        # Entropy-based anomaly detection
        entropy_features = ['cipher_suite_entropy', 'certificate_entropy', 'extension_entropy']
        
        for feature in entropy_features:
            if feature in features_df.columns:
                values = features_df[feature].values
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(values))
                anomaly_scores += z_scores
        
        # Cipher strength anomaly
        if 'cipher_strength' in features_df.columns:
            weak_cipher_mask = features_df['cipher_strength'] < 128
            anomaly_scores[weak_cipher_mask] += 2.0
        
        # PFS anomaly
        if 'perfect_forward_secrecy' in features_df.columns:
            no_pfs_mask = ~features_df['perfect_forward_secrecy']
            anomaly_scores[no_pfs_mask] += 1.0
        
        return anomaly_scores
    
    def detect_tls_anomalies(self, tls_flows: List[Dict], 
                           threshold: float = 2.0) -> List[Dict]:
        """
        Detect TLS-based anomalies in network flows.
        
        Args:
            tls_flows: List of TLS flow data
            threshold: Anomaly score threshold
            
        Returns:
            List of detected anomalies with details
        """
        # Extract features
        features_df = self.extract_handshake_features(tls_flows)
        
        # Calculate anomaly scores
        anomaly_scores = self.calculate_anomaly_scores(features_df)
        
        # Identify anomalies
        anomalies = []
        for i, score in enumerate(anomaly_scores):
            if score > threshold:
                anomaly = {
                    'flow_index': i,
                    'anomaly_score': score,
                    'features': features_df.iloc[i].to_dict(),
                    'flow_data': tls_flows[i] if i < len(tls_flows) else None
                }
                anomalies.append(anomaly)
        
        return anomalies


class EncryptedPayloadAnalyzer:
    """Analyze encrypted payload characteristics for feature extraction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_payload_features(self, encrypted_payloads: List[bytes]) -> Dict[str, float]:
        """
        Extract features from encrypted payload data.
        
        Args:
            encrypted_payloads: List of encrypted payload bytes
            
        Returns:
            Dictionary of payload-based features
        """
        if not encrypted_payloads:
            return self._get_default_payload_features()
        
        features = {}
        
        # Byte distribution entropy
        features['payload_entropy'] = self._calculate_payload_entropy(encrypted_payloads)
        
        # Size distribution features
        sizes = [len(payload) for payload in encrypted_payloads]
        features['size_entropy'] = self._calculate_size_entropy(sizes)
        features['size_variance'] = np.var(sizes) if len(sizes) > 1 else 0.0
        features['size_mean'] = np.mean(sizes)
        
        # Randomness tests
        features['randomness_score'] = self._assess_randomness(encrypted_payloads)
        
        # Pattern detection
        features['pattern_regularity'] = self._detect_patterns(encrypted_payloads)
        
        return features
    
    def _calculate_payload_entropy(self, payloads: List[bytes]) -> float:
        """Calculate entropy of byte distribution across payloads."""
        all_bytes = b''.join(payloads)
        if not all_bytes:
            return 0.0
        
        # Count byte frequencies
        byte_counts = Counter(all_bytes)
        total_bytes = len(all_bytes)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / total_bytes
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_size_entropy(self, sizes: List[int]) -> float:
        """Calculate entropy of payload size distribution."""
        if not sizes:
            return 0.0
        
        size_counts = Counter(sizes)
        total_count = len(sizes)
        
        entropy = 0.0
        for count in size_counts.values():
            probability = count / total_count
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _assess_randomness(self, payloads: List[bytes]) -> float:
        """Assess the randomness of encrypted payloads using statistical tests."""
        if not payloads:
            return 0.0
        
        # Combine all payloads for analysis
        combined_data = b''.join(payloads)
        if len(combined_data) < 100:  # Need sufficient data for tests
            return 0.0
        
        # Convert to numpy array for analysis
        data_array = np.frombuffer(combined_data, dtype=np.uint8)
        
        # Chi-square test for uniformity
        expected_freq = len(data_array) / 256
        observed_freq = np.bincount(data_array, minlength=256)
        chi_square = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
        
        # Normalize chi-square to 0-1 scale (higher = more random)
        randomness_score = 1.0 / (1.0 + chi_square / 1000)
        
        return randomness_score
    
    def _detect_patterns(self, payloads: List[bytes]) -> float:
        """Detect repeating patterns in encrypted payloads."""
        if not payloads:
            return 0.0
        
        pattern_scores = []
        
        for payload in payloads:
            if len(payload) < 16:  # Skip very small payloads
                continue
            
            # Look for repeating byte sequences
            pattern_score = 0.0
            for seq_len in [2, 4, 8]:
                sequences = {}
                for i in range(len(payload) - seq_len + 1):
                    seq = payload[i:i + seq_len]
                    sequences[seq] = sequences.get(seq, 0) + 1
                
                # Calculate pattern regularity
                if sequences:
                    max_count = max(sequences.values())
                    pattern_score += max_count / len(payload)
            
            pattern_scores.append(pattern_score)
        
        return np.mean(pattern_scores) if pattern_scores else 0.0
    
    def _get_default_payload_features(self) -> Dict[str, float]:
        """Return default payload features."""
        return {
            'payload_entropy': 0.0,
            'size_entropy': 0.0,
            'size_variance': 0.0,
            'size_mean': 0.0,
            'randomness_score': 0.0,
            'pattern_regularity': 0.0
        }


# Example usage and testing
if __name__ == "__main__":
    # Example TLS flow data
    sample_tls_flows = [
        {
            'cipher_suites': ['TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384'],
            'certificates': [
                {'size': 1024, 'issuer': 'DigiCert', 'validity_days': 365}
            ],
            'extensions': [
                {'type': 'server_name'}, {'type': 'supported_groups'}
            ],
            'handshake_packets': [
                {'timestamp': 1000.0, 'direction': 'client_to_server', 'size': 512},
                {'timestamp': 1000.1, 'direction': 'server_to_client', 'size': 1024},
            ],
            'tls_version': 'TLSv1.3',
            'cipher_suite': 'TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384',
            'key_exchange': 'ECDHE'
        }
    ]
    
    # Initialize analyzer
    analyzer = TLSEntropyAnalyzer()
    
    # Extract features
    features = analyzer.extract_handshake_features(sample_tls_flows)
    print("Extracted TLS Features:")
    print(features)
    
    # Detect anomalies
    anomalies = analyzer.detect_tls_anomalies(sample_tls_flows)
    print(f"\nDetected {len(anomalies)} anomalies")
