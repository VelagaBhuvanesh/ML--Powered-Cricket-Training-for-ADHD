import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from config_mysql import WINDOW_SIZE, WINDOW_STRIDE, SAMPLE_RATE
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    def __init__(self, sample_rate=100):
        self.sample_rate = sample_rate
        self.channels = ['accx', 'accy', 'accz']
    
    def _zero_crossing_rate(self, data: np.ndarray) -> float:
        """Count how many times signal crosses zero"""
        if len(data) < 2:
            return 0
        return np.sum(np.diff(np.sign(data)) != 0) / len(data)
    
    def _sum_consecutive_diff(self, data: np.ndarray) -> float:
        """Sum of absolute consecutive differences"""
        if len(data) < 2:
            return 0
        return np.sum(np.abs(np.diff(data)))
    
    def _shannon_entropy(self, data: np.ndarray) -> float:
        """Shannon entropy of the signal"""
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
        hist, _ = np.histogram(data_norm, bins=10, range=(0, 1))
        hist = hist / (len(data_norm) + 1e-10)
        return -np.sum(hist * np.log2(hist + 1e-10))
    
    def _spectral_energy(self, data: np.ndarray) -> float:
        """Spectral energy from FFT"""
        fft_vals = np.abs(fft(data))
        return np.sum(fft_vals ** 2) / len(data)
    
    def extract_features(self, window_data: np.ndarray) -> dict:
        """Extract 51 features from a window of sensor data"""
        features = {}
        
        for idx, channel in enumerate(self.channels):
            data = window_data[:, idx]
            
            # Statistical features
            features[f'{channel}_mean'] = np.mean(data)
            features[f'{channel}_std'] = np.std(data)
            features[f'{channel}_variance'] = np.var(data)
            features[f'{channel}_min'] = np.min(data)
            features[f'{channel}_max'] = np.max(data)
            features[f'{channel}_range'] = np.max(data) - np.min(data)
            features[f'{channel}_rms'] = np.sqrt(np.mean(np.square(data)))
            
            # Distribution features
            features[f'{channel}_skewness'] = stats.skew(data)
            features[f'{channel}_kurtosis'] = stats.kurtosis(data)
            features[f'{channel}_p25'] = np.percentile(data, 25)
            features[f'{channel}_p50'] = np.percentile(data, 50)
            features[f'{channel}_p75'] = np.percentile(data, 75)
            features[f'{channel}_iqr'] = features[f'{channel}_p75'] - features[f'{channel}_p25']
            
            # Temporal features
            features[f'{channel}_zcr'] = self._zero_crossing_rate(data)
            features[f'{channel}_scd'] = self._sum_consecutive_diff(data)
            
            # Frequency features
            features[f'{channel}_entropy'] = self._shannon_entropy(data)
            features[f'{channel}_spectral_energy'] = self._spectral_energy(data)
        
        return features
    
    def process_participant(self, sensor_data: pd.DataFrame, label: int, participant_id: int) -> pd.DataFrame:
        """Process all data for a single participant"""
        data = sensor_data[self.channels].values
        window_samples = self.sample_rate * WINDOW_SIZE
        stride = self.sample_rate * WINDOW_STRIDE
        
        windows = []
        window_index = 0
        
        for start in range(0, len(data) - window_samples + 1, stride):
            window = data[start:start + window_samples]
            
            if len(window) == window_samples:
                features = self.extract_features(window)
                features['participant_id'] = participant_id
                features['label'] = label
                features['window_index'] = window_index
                windows.append(features)
                window_index += 1
        
        return pd.DataFrame(windows)