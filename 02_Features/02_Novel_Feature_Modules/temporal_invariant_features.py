"""
Temporal Invariant Features - Novel Feature Module for P22 IDS

This module implements time-invariant features that remain consistent across
different network conditions, making them robust for encrypted traffic classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats, signal
from collections import defaultdict, deque
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler


@dataclass
class TemporalInvariantFeatures:
    """Data class for temporal invariant features."""
    
    # Flow rhythm features
    inter_arrival_cv: float  # Coefficient of variation
    burst_ratio: float
    idle_time_ratio: float
    
    # Size pattern features
    size_sequence_entropy: float
    size_autocorrelation: float
    size_trend_strength: float
    
    # Directional features
    bidirectional_ratio: float
    conversation_turns: int
    dominance_score: float
    
    # Periodicity features
    periodicity_strength: float
    dominant_frequency: float
    spectral_entropy: float
    
    # Statistical invariants
    hurst_exponent: float
    fractal_dimension: float
    long_range_dependence: float


class TemporalInvariantExtractor:
    """
    Extract temporal invariant features from network traffic flows.
    
    These features are designed to be robust against:
    - Network latency variations
    - Bandwidth fluctuations
    - Time-of-day effects
    - Geographic differences
    """
    
    def __init__(self, window_size: int = 100, min_packets: int = 10):
        """
        Initialize temporal invariant feature extractor.
        
        Args:
            window_size: Size of analysis window for temporal features
            min_packets: Minimum packets required for reliable analysis
        """
        self.window_size = window_size
        self.min_packets = min_packets
        self.logger = logging.getLogger(__name__)
    
    def extract_flow_invariants(self, flows: List[Dict]) -> pd.DataFrame:
        """
        Extract temporal invariant features from network flows.
        
        Args:
            flows: List of flow dictionaries with packet-level data
            
        Returns:
            DataFrame with extracted invariant features
        """
        features = []
        
        for flow in flows:
            try:
                flow_features = self._analyze_single_flow(flow)
                features.append(flow_features)
            except Exception as e:
                self.logger.warning(f"Failed to analyze flow: {e}")
                features.append(self._get_default_features())
        
        return pd.DataFrame([vars(f) for f in features])
    
    def _analyze_single_flow(self, flow: Dict) -> TemporalInvariantFeatures:
        """Analyze a single flow for temporal invariant features."""
        
        packets = flow.get('packets', [])
        if len(packets) < self.min_packets:
            return self._get_default_features()
        
        # Extract packet information
        timestamps = np.array([pkt.get('timestamp', 0) for pkt in packets])
        sizes = np.array([pkt.get('size', 0) for pkt in packets])
        directions = [pkt.get('direction', 'unknown') for pkt in packets]
        
        # Calculate inter-arrival times
        inter_arrivals = np.diff(timestamps)
        
        # Flow rhythm features
        ia_cv = self._calculate_coefficient_variation(inter_arrivals)
        burst_ratio = self._calculate_burst_ratio(inter_arrivals)
        idle_ratio = self._calculate_idle_time_ratio(inter_arrivals)
        
        # Size pattern features
        size_entropy = self._calculate_sequence_entropy(sizes)
        size_autocorr = self._calculate_autocorrelation(sizes)
        size_trend = self._calculate_trend_strength(sizes)
        
        # Directional features
        bidir_ratio = self._calculate_bidirectional_ratio(directions)
        conv_turns = self._count_conversation_turns(directions)
        dominance = self._calculate_dominance_score(directions, sizes)
        
        # Periodicity features
        period_strength = self._calculate_periodicity_strength(inter_arrivals)
        dominant_freq = self._find_dominant_frequency(inter_arrivals)
        spectral_ent = self._calculate_spectral_entropy(inter_arrivals)
        
        # Statistical invariants
        hurst_exp = self._calculate_hurst_exponent(inter_arrivals)
        fractal_dim = self._calculate_fractal_dimension(inter_arrivals)
        lrd = self._calculate_long_range_dependence(inter_arrivals)
        
        return TemporalInvariantFeatures(
            inter_arrival_cv=ia_cv,
            burst_ratio=burst_ratio,
            idle_time_ratio=idle_ratio,
            size_sequence_entropy=size_entropy,
            size_autocorrelation=size_autocorr,
            size_trend_strength=size_trend,
            bidirectional_ratio=bidir_ratio,
            conversation_turns=conv_turns,
            dominance_score=dominance,
            periodicity_strength=period_strength,
            dominant_frequency=dominant_freq,
            spectral_entropy=spectral_ent,
            hurst_exponent=hurst_exp,
            fractal_dimension=fractal_dim,
            long_range_dependence=lrd
        )
    
    def _calculate_coefficient_variation(self, values: np.ndarray) -> float:
        """Calculate coefficient of variation (CV = std/mean)."""
        if len(values) == 0 or np.mean(values) == 0:
            return 0.0
        return np.std(values) / np.mean(values)
    
    def _calculate_burst_ratio(self, inter_arrivals: np.ndarray, 
                              threshold_percentile: float = 10) -> float:
        """
        Calculate the ratio of burst periods to total flow duration.
        
        Burst periods are defined as consecutive packets with inter-arrival
        times below the threshold percentile.
        """
        if len(inter_arrivals) == 0:
            return 0.0
        
        threshold = np.percentile(inter_arrivals, threshold_percentile)
        burst_mask = inter_arrivals < threshold
        
        # Find consecutive burst periods
        burst_periods = []
        current_burst = 0
        
        for is_burst in burst_mask:
            if is_burst:
                current_burst += 1
            else:
                if current_burst > 0:
                    burst_periods.append(current_burst)
                current_burst = 0
        
        if current_burst > 0:
            burst_periods.append(current_burst)
        
        total_burst_packets = sum(burst_periods)
        return total_burst_packets / len(inter_arrivals) if len(inter_arrivals) > 0 else 0.0
    
    def _calculate_idle_time_ratio(self, inter_arrivals: np.ndarray,
                                  threshold_percentile: float = 90) -> float:
        """Calculate the ratio of idle time to total flow duration."""
        if len(inter_arrivals) == 0:
            return 0.0
        
        threshold = np.percentile(inter_arrivals, threshold_percentile)
        idle_times = inter_arrivals[inter_arrivals > threshold]
        
        total_idle_time = np.sum(idle_times)
        total_time = np.sum(inter_arrivals)
        
        return total_idle_time / total_time if total_time > 0 else 0.0
    
    def _calculate_sequence_entropy(self, sequence: np.ndarray, bins: int = 10) -> float:
        """Calculate entropy of a sequence using binning."""
        if len(sequence) == 0:
            return 0.0
        
        # Bin the sequence values
        hist, _ = np.histogram(sequence, bins=bins)
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) == 0:
            return 0.0
        
        # Normalize to probabilities
        probabilities = hist / np.sum(hist)
        
        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _calculate_autocorrelation(self, sequence: np.ndarray, max_lag: int = 10) -> float:
        """Calculate maximum autocorrelation within specified lag range."""
        if len(sequence) < max_lag + 1:
            return 0.0
        
        # Normalize sequence
        normalized = (sequence - np.mean(sequence)) / np.std(sequence)
        
        # Calculate autocorrelations
        autocorrs = []
        for lag in range(1, min(max_lag + 1, len(sequence))):
            if len(normalized) > lag:
                corr = np.corrcoef(normalized[:-lag], normalized[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(abs(corr))
        
        return max(autocorrs) if autocorrs else 0.0
    
    def _calculate_trend_strength(self, sequence: np.ndarray) -> float:
        """Calculate the strength of trend in the sequence."""
        if len(sequence) < 3:
            return 0.0
        
        # Linear regression to find trend
        x = np.arange(len(sequence))
        slope, _, r_value, _, _ = stats.linregress(x, sequence)
        
        # Return R-squared as trend strength
        return r_value ** 2 if not np.isnan(r_value) else 0.0
    
    def _calculate_bidirectional_ratio(self, directions: List[str]) -> float:
        """Calculate the ratio of bidirectional communication."""
        if not directions:
            return 0.0
        
        direction_counts = defaultdict(int)
        for direction in directions:
            direction_counts[direction] += 1
        
        if len(direction_counts) < 2:
            return 0.0  # Unidirectional
        
        # Calculate balance between directions
        counts = list(direction_counts.values())
        min_count = min(counts)
        max_count = max(counts)
        
        return min_count / max_count if max_count > 0 else 0.0
    
    def _count_conversation_turns(self, directions: List[str]) -> int:
        """Count the number of conversation turns (direction changes)."""
        if len(directions) < 2:
            return 0
        
        turns = 0
        prev_direction = directions[0]
        
        for direction in directions[1:]:
            if direction != prev_direction:
                turns += 1
                prev_direction = direction
        
        return turns
    
    def _calculate_dominance_score(self, directions: List[str], 
                                  sizes: np.ndarray) -> float:
        """Calculate communication dominance score."""
        if not directions or len(sizes) == 0:
            return 0.0
        
        direction_bytes = defaultdict(int)
        for direction, size in zip(directions, sizes):
            direction_bytes[direction] += size
        
        if len(direction_bytes) < 2:
            return 1.0  # Complete dominance
        
        total_bytes = sum(direction_bytes.values())
        if total_bytes == 0:
            return 0.0
        
        # Calculate Gini coefficient for dominance
        sorted_bytes = sorted(direction_bytes.values())
        n = len(sorted_bytes)
        
        gini = (2 * sum((i + 1) * bytes_val for i, bytes_val in enumerate(sorted_bytes))) / (n * total_bytes) - (n + 1) / n
        
        return gini
    
    def _calculate_periodicity_strength(self, inter_arrivals: np.ndarray) -> float:
        """Calculate the strength of periodic patterns."""
        if len(inter_arrivals) < 10:
            return 0.0
        
        # Use autocorrelation to detect periodicity
        autocorr = np.correlate(inter_arrivals, inter_arrivals, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Normalize
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
        
        # Find peaks (excluding the first one at lag 0)
        if len(autocorr) > 1:
            peaks = signal.find_peaks(autocorr[1:], height=0.1)[0]
            return max(autocorr[peaks + 1]) if len(peaks) > 0 else 0.0
        
        return 0.0
    
    def _find_dominant_frequency(self, inter_arrivals: np.ndarray) -> float:
        """Find the dominant frequency in the inter-arrival time series."""
        if len(inter_arrivals) < 10:
            return 0.0
        
        # Compute power spectral density
        freqs, psd = signal.periodogram(inter_arrivals)
        
        # Find dominant frequency (excluding DC component)
        if len(psd) > 1:
            dominant_idx = np.argmax(psd[1:]) + 1
            return freqs[dominant_idx]
        
        return 0.0
    
    def _calculate_spectral_entropy(self, inter_arrivals: np.ndarray) -> float:
        """Calculate entropy of the power spectral density."""
        if len(inter_arrivals) < 10:
            return 0.0
        
        # Compute power spectral density
        _, psd = signal.periodogram(inter_arrivals)
        
        # Normalize to probabilities
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        
        # Calculate entropy
        entropy = 0.0
        for p in psd_norm:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_hurst_exponent(self, time_series: np.ndarray) -> float:
        """
        Calculate Hurst exponent using R/S analysis.
        
        Hurst exponent indicates long-term memory in time series:
        - H < 0.5: Anti-persistent (mean-reverting)
        - H = 0.5: Random walk
        - H > 0.5: Persistent (trending)
        """
        if len(time_series) < 10:
            return 0.5  # Default to random walk
        
        try:
            # Remove mean
            Y = np.cumsum(time_series - np.mean(time_series))
            
            # Calculate range
            R = np.max(Y) - np.min(Y)
            
            # Calculate standard deviation
            S = np.std(time_series)
            
            if S == 0:
                return 0.5
            
            # R/S ratio
            rs_ratio = R / S
            
            # Hurst exponent approximation
            n = len(time_series)
            hurst = np.log(rs_ratio) / np.log(n)
            
            # Clamp to reasonable range
            return max(0.0, min(1.0, hurst))
            
        except Exception:
            return 0.5
    
    def _calculate_fractal_dimension(self, time_series: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method."""
        if len(time_series) < 10:
            return 1.0
        
        try:
            # Normalize time series
            normalized = (time_series - np.min(time_series)) / (np.max(time_series) - np.min(time_series) + 1e-10)
            
            # Box-counting algorithm
            scales = np.logspace(0.01, 1, num=10)
            counts = []
            
            for scale in scales:
                # Grid size
                grid_size = int(1 / scale)
                if grid_size < 1:
                    grid_size = 1
                
                # Count boxes containing data points
                boxes = set()
                for i, value in enumerate(normalized):
                    box_x = int(i / len(normalized) * grid_size)
                    box_y = int(value * grid_size)
                    boxes.add((box_x, box_y))
                
                counts.append(len(boxes))
            
            # Linear regression in log-log space
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            
            slope, _, _, _, _ = stats.linregress(log_scales, log_counts)
            
            # Fractal dimension
            fractal_dim = -slope
            
            return max(1.0, min(2.0, fractal_dim))
            
        except Exception:
            return 1.0
    
    def _calculate_long_range_dependence(self, time_series: np.ndarray) -> float:
        """Calculate long-range dependence using detrended fluctuation analysis."""
        if len(time_series) < 20:
            return 0.0
        
        try:
            # Integration
            y = np.cumsum(time_series - np.mean(time_series))
            
            # Window sizes
            window_sizes = np.unique(np.logspace(0.5, np.log10(len(y) // 4), num=10).astype(int))
            
            fluctuations = []
            
            for window_size in window_sizes:
                if window_size < 4:
                    continue
                
                # Divide into windows
                n_windows = len(y) // window_size
                local_trends = []
                
                for i in range(n_windows):
                    start_idx = i * window_size
                    end_idx = (i + 1) * window_size
                    window_data = y[start_idx:end_idx]
                    
                    # Linear detrending
                    x = np.arange(len(window_data))
                    coeffs = np.polyfit(x, window_data, 1)
                    trend = np.polyval(coeffs, x)
                    
                    # Calculate fluctuation
                    detrended = window_data - trend
                    local_trends.append(np.sqrt(np.mean(detrended ** 2)))
                
                if local_trends:
                    fluctuations.append(np.mean(local_trends))
            
            if len(fluctuations) < 3:
                return 0.0
            
            # Power law fitting
            log_windows = np.log(window_sizes[:len(fluctuations)])
            log_fluctuations = np.log(fluctuations)
            
            slope, _, r_value, _, _ = stats.linregress(log_windows, log_fluctuations)
            
            # Long-range dependence parameter
            lrd = slope - 0.5  # Subtract 0.5 for white noise baseline
            
            return max(0.0, min(1.0, abs(lrd)))
            
        except Exception:
            return 0.0
    
    def _get_default_features(self) -> TemporalInvariantFeatures:
        """Return default features for failed analysis."""
        return TemporalInvariantFeatures(
            inter_arrival_cv=0.0,
            burst_ratio=0.0,
            idle_time_ratio=0.0,
            size_sequence_entropy=0.0,
            size_autocorrelation=0.0,
            size_trend_strength=0.0,
            bidirectional_ratio=0.0,
            conversation_turns=0,
            dominance_score=0.0,
            periodicity_strength=0.0,
            dominant_frequency=0.0,
            spectral_entropy=0.0,
            hurst_exponent=0.5,
            fractal_dimension=1.0,
            long_range_dependence=0.0
        )
    
    def calculate_invariance_scores(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate invariance scores for features.
        
        Higher scores indicate more invariant (stable) features.
        """
        invariance_scores = np.zeros(len(features_df))
        
        # Features that should be stable across conditions
        stable_features = [
            'size_sequence_entropy',
            'size_autocorrelation', 
            'bidirectional_ratio',
            'conversation_turns',
            'periodicity_strength',
            'hurst_exponent',
            'fractal_dimension'
        ]
        
        for feature in stable_features:
            if feature in features_df.columns:
                values = features_df[feature].values
                # Lower coefficient of variation indicates higher invariance
                cv = np.std(values) / (np.mean(values) + 1e-10)
                invariance_scores += 1.0 / (1.0 + cv)
        
        return invariance_scores / len(stable_features)


class FlowRhythmAnalyzer:
    """Specialized analyzer for flow rhythm patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_rhythm_features(self, inter_arrivals: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive rhythm features from inter-arrival times."""
        
        if len(inter_arrivals) == 0:
            return self._get_default_rhythm_features()
        
        features = {}
        
        # Basic rhythm statistics
        features['rhythm_mean'] = np.mean(inter_arrivals)
        features['rhythm_std'] = np.std(inter_arrivals)
        features['rhythm_cv'] = features['rhythm_std'] / (features['rhythm_mean'] + 1e-10)
        
        # Rhythm regularity
        features['rhythm_regularity'] = self._calculate_rhythm_regularity(inter_arrivals)
        
        # Burst characteristics
        features['burst_intensity'] = self._calculate_burst_intensity(inter_arrivals)
        features['burst_frequency'] = self._calculate_burst_frequency(inter_arrivals)
        
        # Rhythm complexity
        features['rhythm_complexity'] = self._calculate_rhythm_complexity(inter_arrivals)
        
        return features
    
    def _calculate_rhythm_regularity(self, inter_arrivals: np.ndarray) -> float:
        """Calculate how regular the rhythm is."""
        if len(inter_arrivals) < 3:
            return 0.0
        
        # Use coefficient of variation as inverse regularity measure
        cv = np.std(inter_arrivals) / (np.mean(inter_arrivals) + 1e-10)
        regularity = 1.0 / (1.0 + cv)
        
        return regularity
    
    def _calculate_burst_intensity(self, inter_arrivals: np.ndarray) -> float:
        """Calculate the intensity of burst periods."""
        if len(inter_arrivals) == 0:
            return 0.0
        
        # Define burst threshold as 25th percentile
        threshold = np.percentile(inter_arrivals, 25)
        burst_times = inter_arrivals[inter_arrivals < threshold]
        
        if len(burst_times) == 0:
            return 0.0
        
        # Intensity is inverse of mean burst inter-arrival time
        mean_burst_time = np.mean(burst_times)
        intensity = 1.0 / (mean_burst_time + 1e-10)
        
        return intensity
    
    def _calculate_burst_frequency(self, inter_arrivals: np.ndarray) -> float:
        """Calculate how frequently bursts occur."""
        if len(inter_arrivals) == 0:
            return 0.0
        
        threshold = np.percentile(inter_arrivals, 25)
        burst_mask = inter_arrivals < threshold
        
        # Count burst periods
        burst_periods = 0
        in_burst = False
        
        for is_burst in burst_mask:
            if is_burst and not in_burst:
                burst_periods += 1
                in_burst = True
            elif not is_burst:
                in_burst = False
        
        # Frequency as bursts per total packets
        frequency = burst_periods / len(inter_arrivals)
        
        return frequency
    
    def _calculate_rhythm_complexity(self, inter_arrivals: np.ndarray) -> float:
        """Calculate the complexity of the rhythm pattern."""
        if len(inter_arrivals) < 10:
            return 0.0
        
        # Use approximate entropy to measure complexity
        m = 2  # Pattern length
        r = 0.2 * np.std(inter_arrivals)  # Tolerance
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([inter_arrivals[i:i + m] for i in range(len(inter_arrivals) - m + 1)])
            C = np.zeros(len(patterns))
            
            for i in range(len(patterns)):
                template_i = patterns[i]
                for j in range(len(patterns)):
                    if _maxdist(template_i, patterns[j], m) <= r:
                        C[i] += 1.0
            
            phi = np.mean(np.log(C / len(patterns)))
            return phi
        
        try:
            complexity = _phi(m) - _phi(m + 1)
            return max(0.0, complexity)
        except Exception:
            return 0.0
    
    def _get_default_rhythm_features(self) -> Dict[str, float]:
        """Return default rhythm features."""
        return {
            'rhythm_mean': 0.0,
            'rhythm_std': 0.0,
            'rhythm_cv': 0.0,
            'rhythm_regularity': 0.0,
            'burst_intensity': 0.0,
            'burst_frequency': 0.0,
            'rhythm_complexity': 0.0
        }


# Example usage
if __name__ == "__main__":
    # Example flow data
    sample_flows = [
        {
            'packets': [
                {'timestamp': 1000.0, 'size': 64, 'direction': 'client_to_server'},
                {'timestamp': 1000.1, 'size': 1024, 'direction': 'server_to_client'},
                {'timestamp': 1000.15, 'size': 128, 'direction': 'client_to_server'},
                {'timestamp': 1000.3, 'size': 512, 'direction': 'server_to_client'},
                {'timestamp': 1000.35, 'size': 256, 'direction': 'client_to_server'},
            ]
        }
    ]
    
    # Initialize extractor
    extractor = TemporalInvariantExtractor()
    
    # Extract features
    features = extractor.extract_flow_invariants(sample_flows)
    print("Extracted Temporal Invariant Features:")
    print(features)
    
    # Calculate invariance scores
    invariance_scores = extractor.calculate_invariance_scores(features)
    print(f"\nInvariance Scores: {invariance_scores}")
