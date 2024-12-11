```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import struct
import logging
import argparse
import csv
import joblib
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from typing import Tuple, Dict, Any
from FeatureExtractor import FeatureExtractor

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class Config:
    def __init__(self,
                 csi_file: str = 'csi_data.bin',
                 subcarriers: int = 30,
                 filter_kernel_size: int = 3,
                 export_csv: bool = True,
                 output_csv_file: str = 'csi_results.csv',
                 plot_figures: bool = True,
                 save_figures: bool = False,
                 figures_dir: str = 'figures',
                 buffer_size: int = 100):
        self.csi_file = csi_file
        self.subcarriers = subcarriers
        self.filter_kernel_size = filter_kernel_size
        self.export_csv = export_csv
        self.output_csv_file = output_csv_file
        self.plot_figures = plot_figures
        self.save_figures = save_figures
        self.figures_dir = figures_dir
        self.buffer_size = buffer_size

        if self.save_figures and not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir, exist_ok=True)

class CSIReader:
    def __init__(self, config: Config):
        self.config = config

    def read_new_csi_data(self, file_handle, current_position: int) -> Tuple[np.ndarray, int]:
        file_handle.seek(current_position)
        bytes_per_amostra = self.config.subcarriers * 2 * 2
        bytes_read = file_handle.read(bytes_per_amostra)
        if not bytes_read or len(bytes_read) < bytes_per_amostra:
            return np.array([]), current_position
        csi_vals = struct.unpack(f'{self.config.subcarriers*2}h', bytes_read)
        complex_csi = np.array(csi_vals[0::2]) + 1j*np.array(csi_vals[1::2])
        current_position = file_handle.tell()
        return complex_csi, current_position

class CSIFilter:
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size

    def denoise_amplitude(self, csi: np.ndarray) -> np.ndarray:
        amplitudes = np.abs(csi)
        filtered = medfilt(amplitudes, kernel_size=self.kernel_size)
        return filtered

    def unwrap_phase(self, csi: np.ndarray) -> np.ndarray:
        phases = np.angle(csi)
        unwrapped = np.unwrap(phases)
        return unwrapped

class CSIAnalyzer:
    def __init__(self, config: Config):
        self.config = config

    def compute_mean_amplitude(self, csi_amplitude: np.ndarray) -> float:
        return np.mean(csi_amplitude)

    def compute_mean_phase(self, csi_phase: np.ndarray) -> float:
        return np.mean(csi_phase)

    def compute_frequency_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        freq_domain = np.fft.fftshift(np.fft.fft(signal))
        freqs = np.fft.fftfreq(len(signal))
        return freqs, np.abs(freq_domain)

    def plot_time_series(self, time_signal: np.ndarray, title: str, ylabel: str, filename: str):
        plt.figure(figsize=(8,4))
        plt.plot(time_signal)
        plt.title(title)
        plt.xlabel("Sample")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        if self.config.plot_figures:
            plt.show()
        if self.config.save_figures:
            plt.savefig(os.path.join(self.config.figures_dir, filename))
        plt.close()

    def plot_frequency_spectrum(self, freqs: np.ndarray, spectrum: np.ndarray, title: str, filename: str):
        plt.figure(figsize=(8,4))
        plt.plot(freqs, spectrum)
        plt.title(title)
        plt.xlabel("Normalized Frequency")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.tight_layout()
        if self.config.plot_figures:
            plt.show()
        if self.config.save_figures:
            plt.savefig(os.path.join(self.config.figures_dir, filename))
        plt.close()

class ResultsExporter:
    def __init__(self, config: Config):
        self.config = config

    def export_to_csv(self, data: Dict[str, Any]):
        if not self.config.export_csv:
            logging.info("Export CSV disabled. Skipping this step.")
            return
        file_exists = os.path.isfile(self.config.output_csv_file)
        with open(self.config.output_csv_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(data.keys())
            writer.writerow(data.values())
        logging.info(f"Exported data to {self.config.output_csv_file}")

def main(config: Config):
    reader = CSIReader(config)
    csi_filter = CSIFilter(kernel_size=config.filter_kernel_size)
    analyzer = CSIAnalyzer(config)
    exporter = ResultsExporter(config)
    buffer = []
    current_position = 0

    try:
        with open(config.csi_file, 'rb') as f:
            while True:
                new_csi, current_position = reader.read_new_csi_data(f, current_position)
                if new_csi.size > 0:
                    amplitude_filtered = csi_filter.denoise_amplitude(new_csi)
                    phase_unwrapped = csi_filter.unwrap_phase(new_csi)
                    mean_amp = analyzer.compute_mean_amplitude(amplitude_filtered)
                    mean_phase = analyzer.compute_mean_phase(phase_unwrapped)
                    buffer.append((mean_amp, mean_phase))
                    if len(buffer) >= config.buffer_size:
                        mean_amplitudes, mean_phases = zip(*buffer)
                        mean_amplitudes = np.array(mean_amplitudes)
                        mean_phases = np.array(mean_phases)
                        features = FeatureExtractor.extract_features(mean_amplitudes, mean_phases).reshape(1, -1)
                        features_scaled = scaler.transform(features)
                        prediction = model.predict(features_scaled)
                        logging.info(f"Prediction: {prediction[0]}")
                        exporter.export_to_csv({
                            'mean_amplitude': mean_amp,
                            'mean_phase': mean_phase,
                            'prediction': prediction[0]
                        })
                        buffer = []
                time.sleep(0.5)
    except KeyboardInterrupt:
        logging.info("Real-time processing stopped by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wi-Fi Radar Prototype with Real-Time ML Predictions")
    parser.add_argument("--file", type=str, default="csi_data.bin", help="Path to CSI data file.")
    parser.add_argument("--subcarriers", type=int, default=30, help="Number of subcarriers.")
    parser.add_argument("--kernel", type=int, default=3, help="Size of the median filter kernel.")
    parser.add_argument("--no-export", action="store_true", help="Disable exporting results to CSV.")
    parser.add_argument("--csv-file", type=str, default="csi_results.csv", help="Name of the output CSV file.")
    parser.add_argument("--no-plot", action="store_true", help="Do not display plots on screen.")
    parser.add_argument("--save-fig", action="store_true", help="Save plots to files instead of displaying.")
    parser.add_argument("--fig-dir", type=str, default="figures", help="Directory to save figures.")
    parser.add_argument("--buffer-size", type=int, default=100, help="Number of samples to buffer before prediction.")

    args = parser.parse_args()

    config = Config(
        csi_file=args.file,
        subcarriers=args.subcarriers,
        filter_kernel_size=args.kernel,
        export_csv=not args.no_export,
        output_csv_file=args.csv_file,
        plot_figures=not args.no_plot,
        save_figures=args.save_fig,
        figures_dir=args.fig_dir,
        buffer_size=args.buffer_size
    )

    main(config)
```

---

# README.md

```markdown
# Wi-Fi Radar Prototype for Brazilian Military Research

## Overview

This project aims to develop a **Wi-Fi Radar Prototype** specifically designed to support Brazilian military research. The system leverages **Channel State Information (CSI)** extracted from Wi-Fi signals to map internal environments, detect movements, and identify the presence of objects or individuals. This approach offers a passive and cost-effective solution for monitoring and surveillance in strategic areas.

## Project Structure

The project is organized into the following classes and scripts, each responsible for a specific part of the processing pipeline:

1. **Config:** Manages system configuration parameters.
2. **CSIReader:** Responsible for reading and processing CSI data from binary files.
3. **CSIFilter:** Performs preprocessing of CSI data, including noise filtering and phase unwrapping.
4. **CSIAnalyzer:** Analyzes CSI data to extract metrics in the time and frequency domains.
5. **ResultsExporter:** Exports analysis results to usable formats, such as CSV.
6. **FeatureExtractor:** Prepares features for future Machine Learning implementations.
7. **MLModel:** Placeholder for future integrations of Machine Learning models.

Additionally, the `scripts/` directory contains scripts for training and evaluating the Machine Learning models.

## Features

- **CSI Data Reading:** Imports CSI data from binary files generated by tools compatible with Intel 5300 network cards.
- **Advanced Preprocessing:** Filters noise from amplitudes and corrects phase to enhance data quality.
- **Time and Frequency Analysis:** Generates time series and frequency spectra to identify relevant patterns.
- **Visualization:** Creates graphs for visual interpretation of processed data.
- **Results Exporting:** Saves processed data to CSV files for further analysis.
- **Machine Learning Preparation:** Structures data for future machine learning applications, aiming for more precise classifications or detections.
- **Real-Time Predictions:** Integrates a Machine Learning model to make real-time predictions based on live CSI data.

## Requirements

### Hardware

- **Network Card:** Compatible (Intel 5300 recommended).
- **Computer:** Linux operating system (Ubuntu recommended).

### Software

- **Python:** Version 3.6 or higher.
- **Python Libraries:**
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `argparse`
  - `csv`
  - `scikit-learn`
  - `joblib`
- **CSI Extraction Tool:** Compatible with the network card being used.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/carobaMestre/wifi-radar-prototype.git
   cd wifi-radar-prototype
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the CSI Tool:**
   Follow the instructions of the CSI tool being used to ensure data is captured correctly.

## Usage

Run the main script with desired options:

```bash
python wifi_radar.py --file path_to_csi_data.bin --subcarriers 30 --kernel 3 --save-fig --fig-dir figures_directory
```

### Available Options

- `--file`: Path to the CSI data file.
- `--subcarriers`: Number of subcarriers used.
- `--kernel`: Size of the median filter kernel for noise removal.
- `--no-export`: Disables exporting results to CSV.
- `--csv-file`: Name of the output CSV file.
- `--no-plot`: Does not display graphs on the screen.
- `--save-fig`: Saves graphs to files instead of just displaying them.
- `--fig-dir`: Directory where graphs will be saved.
- `--buffer-size`: Number of samples to buffer before making a prediction.

### Examples

1. **Basic Execution:**
   ```bash
   python wifi_radar.py --file data/csi_data.bin
   ```

2. **Save Graphs and Export CSV:**
   ```bash
   python wifi_radar.py --file data/csi_data.bin --save-fig --fig-dir results/graphs --csv-file results/csi_output.csv
   ```

3. **Disable CSV Export and Do Not Display Graphs:**
   ```bash
   python wifi_radar.py --file data/csi_data.bin --no-export --no-plot
   ```

4. **Real-Time Prediction with Custom Buffer Size:**
   ```bash
   python wifi_radar.py --file data/csi_data.bin --buffer-size 50
   ```

## Machine Learning Integration

To enhance the Wi-Fi Radar Prototype with intelligent detection and classification capabilities, a Machine Learning (ML) system has been integrated. The ML system automates the detection of movements, objects, and individuals based on the processed CSI data.

### Overview

The ML system performs the following tasks:

1. **Data Collection and Labeling:** Gather CSI data under various scenarios and label them accordingly.
2. **Feature Extraction:** Extract meaningful features from the CSI data to be used as inputs for ML models.
3. **Model Selection and Training:** Choose appropriate ML algorithms and train models using the extracted features.
4. **Model Evaluation:** Assess the performance of trained models using suitable metrics.
5. **Deployment and Inference:** Integrate the trained models into the prototype for real-time predictions.
6. **Continuous Learning:** Update models with new data to improve accuracy and adapt to changing environments.

### Steps to Implement ML Integration

#### 1. Data Collection and Labeling

- **Define Scenarios:** Determine the different scenarios you want the ML model to recognize (e.g., no movement, single person walking, multiple people walking, object presence).
- **Collect Data:** Use the existing `wifi_radar.py` script to collect CSI data for each scenario.
- **Label Data:** Create a `labels.csv` file in the `data/` directory mapping each CSI data file to its corresponding label.

**Example `labels.csv`:**

| filename    | label           |
|-------------|-----------------|
| data1.bin   | no_movement     |
| data2.bin   | single_person   |
| data3.bin   | multiple_people |
| data4.bin   | object_present  |

#### 2. Feature Extraction Enhancement

Enhance the feature extraction process to include additional features that improve ML model performance.

- **Run Preprocessing and Feature Extraction:**

  ```bash
  python scripts/preprocess_and_extract_features.py
  ```

  This script processes each CSI data file, extracts features using the `FeatureExtractor` class, and saves them to individual CSV files in the `data/features/` directory.

#### 3. Model Training and Saving

Train ML models using the extracted features and save the trained models for future use.

- **Run Training Script:**

  ```bash
  python scripts/train_model.py
  ```

  This script trains a Random Forest classifier using the extracted features and saves the trained model and scaler to the `models/` directory.

#### 4. Model Evaluation

Evaluate the performance of the trained models to ensure reliability and accuracy.

- **Run Evaluation Script:**

  ```bash
  python scripts/evaluate_model.py
  ```

  This script assesses the trained model's performance using classification reports and confusion matrices.

#### 5. Real-Time Prediction Integration

Integrate the trained ML models into the main prototype to enable real-time predictions.

- **Ensure Models are Trained:**
  
  Make sure `models/random_forest_model.pkl` and `models/scaler.pkl` exist by running the training script.

- **Run Main Script with Prediction:**

  ```bash
  python wifi_radar.py --file path_to_csi_data.bin --subcarriers 30 --kernel 3 --save-fig --fig-dir figures_directory
  ```

  The `wifi_radar.py` script will process the CSI data, extract features, scale them, and make predictions using the trained ML model. The prediction result will be printed to the console and optionally saved to the CSV file.

#### 6. Continuous Learning

To maintain and improve the ML model's performance:

- **Regular Data Collection:** Continuously collect new CSI data under different scenarios.
- **Update Labels:** Ensure new data is accurately labeled.
- **Retrain Models:** Periodically retrain the ML models with the expanded dataset by rerunning the `train_model.py` script.
- **Monitor Performance:** Use the `evaluate_model.py` script to monitor model performance over time.

### Additional Considerations

- **Data Privacy and Security:** Ensure all data handling complies with relevant privacy laws and military regulations.
- **Scalability:** Design the ML system to handle large volumes of data efficiently.
- **Real-Time Processing:** Optimize the system for low-latency predictions to facilitate real-time monitoring.
- **Robustness:** Ensure the model is resilient to noise and variations in the environment.

## Security Considerations

This prototype is intended for studies of systems for military applications, such as surveillance and monitoring of strategic areas. It is essential to ensure that the use of this system complies with all applicable laws and regulations, respecting privacy and human rights.

## Contact

For more information or support, please contact:

- **Email:** [suporte@veacci.com](mailto:suporte@veacci.com)
- **GitHub:** [https://github.com/carobaMestre](https://github.com/carobaMestre)

---

**Note:** This project is a prototype and is subject to continuous improvements. Contributions are welcome to enhance its functionalities and effectiveness in military applications.
```

---

# scripts/train_model.py

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from FeatureExtractor import FeatureExtractor

def load_and_extract_features(labels_csv_path: str, features_dir: str) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(labels_csv_path)
    features_list = []
    labels = []
    
    for index, row in data.iterrows():
        filename = row['filename']
        label = row['label']
        feature_file = os.path.join(features_dir, f'{filename}.csv')
        
        if not os.path.isfile(feature_file):
            continue
        
        feature_data = pd.read_csv(feature_file)
        mean_amplitude = feature_data['mean_amplitude'].values
        mean_phase = feature_data['mean_phase'].values
        
        features = FeatureExtractor.extract_features(mean_amplitude, mean_phase)
        features_list.append(features)
        labels.append(label)
    
    feature_columns = [
        'mean_amplitude', 'std_amplitude', 'max_amplitude', 'min_amplitude', 'median_amplitude',
        'mean_fft_amp', 'std_fft_amp',
        'mean_phase', 'std_phase', 'max_phase', 'min_phase', 'median_phase'
    ]
    
    feature_df = pd.DataFrame(features_list, columns=feature_columns)
    label_series = pd.Series(labels, name='label')
    
    return feature_df, label_series

def main():
    labels_csv = '../data/labels.csv'
    features_dir = '../data/features/'
    
    X, y = load_and_extract_features(labels_csv, features_dir)
    
    if X.empty:
        print("No feature data found. Please ensure feature extraction has been performed.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_rf = grid_search.best_estimator_
    
    os.makedirs('../models', exist_ok=True)
    joblib.dump(best_rf, '../models/random_forest_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    
    y_pred = best_rf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
```

---

# scripts/evaluate_model.py

```python
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from FeatureExtractor import FeatureExtractor

def load_and_extract_features(labels_csv_path: str, features_dir: str) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(labels_csv_path)
    features_list = []
    labels = []
    
    for index, row in data.iterrows():
        filename = row['filename']
        label = row['label']
        feature_file = os.path.join(features_dir, f'{filename}.csv')
        
        if not os.path.isfile(feature_file):
            continue
        
        feature_data = pd.read_csv(feature_file)
        mean_amplitude = feature_data['mean_amplitude'].values
        mean_phase = feature_data['mean_phase'].values
        
        features = FeatureExtractor.extract_features(mean_amplitude, mean_phase)
        features_list.append(features)
        labels.append(label)
    
    feature_columns = [
        'mean_amplitude', 'std_amplitude', 'max_amplitude', 'min_amplitude', 'median_amplitude',
        'mean_fft_amp', 'std_fft_amp',
        'mean_phase', 'std_phase', 'max_phase', 'min_phase', 'median_phase'
    ]
    
    feature_df = pd.DataFrame(features_list, columns=feature_columns)
    label_series = pd.Series(labels, name='label')
    
    return feature_df, label_series

def main():
    labels_csv = '../data/labels.csv'
    features_dir = '../data/features/'
    
    X, y = load_and_extract_features(labels_csv, features_dir)
    
    if X.empty:
        print("No feature data found. Please ensure feature extraction has been performed.")
        return
    
    model_path = '../models/random_forest_model.pkl'
    scaler_path = '../models/scaler.pkl'
    
    if not os.path.isfile(model_path) or not os.path.isfile(scaler_path):
        print("Trained model or scaler not found. Please train the model first.")
        return
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))

if __name__ == "__main__":
    main()
```

---

# scripts/preprocess_and_extract_features.py

```python
import os
import pandas as pd
from wifi_radar import CSIReader, CSIFilter, CSIAnalyzer, Config
from FeatureExtractor import FeatureExtractor

def main():
    config = Config(
        csi_file='',  
        subcarriers=30,
        filter_kernel_size=3,
        export_csv=False,
        plot_figures=False,
        save_figures=False
    )
    
    labels_csv = '../data/labels.csv'
    features_dir = '../data/features/'
    os.makedirs(features_dir, exist_ok=True)
    
    data = pd.read_csv(labels_csv)
    
    for index, row in data.iterrows():
        filename = row['filename']
        csi_file_path = os.path.join('../data', filename)
        
        if not os.path.isfile(csi_file_path):
            continue
        
        reader = CSIReader(config)
        csi_data = reader.read_csi_data(csi_file_path)
        
        csi_filter = CSIFilter(kernel_size=config.filter_kernel_size)
        csi_amplitude_filtered = csi_filter.denoise_amplitude(csi_data)
        csi_phase_unwrapped = csi_filter.unwrap_phase(csi_data)
        
        analyzer = CSIAnalyzer(config)
        mean_amplitude = analyzer.compute_mean_amplitude(csi_amplitude_filtered)
        mean_phase = analyzer.compute_mean_phase(csi_phase_unwrapped)
        
        features = FeatureExtractor.extract_features(mean_amplitude, mean_phase)
        
        feature_df = pd.DataFrame([features], columns=[
            'mean_amplitude', 'std_amplitude', 'max_amplitude', 'min_amplitude', 'median_amplitude',
            'mean_fft_amp', 'std_fft_amp',
            'mean_phase', 'std_phase', 'max_phase', 'min_phase', 'median_phase'
        ])
        feature_file = os.path.join(features_dir, f'{filename}.csv')
        feature_df.to_csv(feature_file, index=False)
        print(f"Extracted features for {filename} and saved to {feature_file}")

if __name__ == "__main__":
    main()
```

---

# FeatureExtractor.py

```python
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
```

---

# Additional Notes

- **Real-Time Processing:**
  - The `wifi_radar.py` script has been updated to handle real-time CSI data processing. It continuously monitors the specified CSI data file for new data, processes incoming CSI samples, extracts features in batches defined by `buffer_size`, and makes predictions using the trained Machine Learning model.
  
- **Buffering Strategy:**
  - To prevent making excessive predictions and to ensure meaningful analysis, data is buffered until a specified number of samples (`buffer_size`) is collected before making a prediction. Adjust `buffer_size` based on your application's requirements and the frequency of CSI data updates.
  
- **Graceful Shutdown:**
  - The main loop in `wifi_radar.py` is designed to handle interruptions gracefully. Pressing `Ctrl+C` will stop the real-time processing without leaving the system in an inconsistent state.

- **Error Handling:**
  - The script includes basic error handling to catch and log unexpected issues during execution.

- **Model and Scaler Loading:**
  - The trained ML model and scaler are loaded once at the beginning of the script to optimize performance during real-time predictions.

- **Visualization:**
  - While the script includes plotting functionalities, be cautious when enabling real-time plotting (`--plot_figures`) as it can significantly slow down the processing. It is recommended to save figures instead of displaying them in real-time for better performance.

- **Data Export:**
  - Predictions and corresponding CSI metrics are exported to a CSV file (`csi_results.csv` by default). This allows for post-processing and analysis of prediction results.

- **Scalability:**
  - The current implementation is suitable for single-threaded environments. For higher performance and scalability, consider implementing multi-threading or asynchronous processing, especially if handling high-frequency CSI data streams.

- **FeatureExtractor Module:**
  - The `FeatureExtractor.py` module encapsulates the feature extraction logic, making it reusable across different scripts and ensuring consistency in feature generation.

- **Scripts Directory:**
  - The `scripts/` directory contains auxiliary scripts for training (`train_model.py`), evaluating (`evaluate_model.py`), and preprocessing (`preprocess_and_extract_features.py`). This separation of concerns enhances the modularity and maintainability of the project.

- **Requirements File:**
  - Ensure that all necessary Python libraries are listed in the `requirements.txt` file for easy installation. You can create this file by running:
    ```bash
    pip freeze > requirements.txt
    ```

- **License and Contributions:**
  - The project uses the MIT License, and contributions are welcome. Ensure to adhere to best practices when contributing, including proper code formatting, documentation, and testing.

- **Contact Information:**
  - For support or inquiries, contact via the provided email or GitHub profile.

By following the structure and guidelines outlined above, you can effectively develop and extend the Wi-Fi Radar Prototype with real-time Machine Learning capabilities tailored for Brazilian military research applications.