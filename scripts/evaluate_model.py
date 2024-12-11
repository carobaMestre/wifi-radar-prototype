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
