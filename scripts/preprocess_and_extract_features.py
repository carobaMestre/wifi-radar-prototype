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
