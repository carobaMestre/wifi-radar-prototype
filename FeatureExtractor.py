import numpy as np

class FeatureExtractor:
    @staticmethod
    def extract_features(amplitude_signal: np.ndarray, phase_signal: np.ndarray) -> np.ndarray:
        features = []
        # Amplitude Features
        features.append(np.mean(amplitude_signal))
        features.append(np.std(amplitude_signal))
        features.append(np.max(amplitude_signal))
        features.append(np.min(amplitude_signal))
        features.append(np.median(amplitude_signal))
        # Frequency Features
        fft_features = np.fft.fft(amplitude_signal)
        features.append(np.abs(np.mean(fft_features)))
        features.append(np.abs(np.std(fft_features)))
        # Phase Features
        features.append(np.mean(phase_signal))
        features.append(np.std(phase_signal))
        features.append(np.max(phase_signal))
        features.append(np.min(phase_signal))
        features.append(np.median(phase_signal))
        return np.array(features)
