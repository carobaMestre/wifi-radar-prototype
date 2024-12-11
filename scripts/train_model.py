import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from FeatureExtractor import FeatureExtractor

def load_and_extract_features(labels_csv_path, features_dir):
    data = pd.read_csv(labels_csv_path)
    features_list = []
    labels = []
    
    for index, row in data.iterrows():
        filename = row['filename']
        label = row['label']
        feature_file = os.path.join(features_dir, f'{filename}.csv')  # Adjust path as needed
        
        # Load pre-extracted features
        feature_data = pd.read_csv(feature_file)
        mean_amplitude = feature_data['mean_amplitude'].values
        mean_phase = feature_data['mean_phase'].values
        
        features = FeatureExtractor.extract_features(mean_amplitude, mean_phase)
        features_list.append(features)
        labels.append(label)
    
    return pd.DataFrame(features_list), pd.Series(labels)

def main():
    labels_csv = '../data/labels.csv'  # Adjust relative path as needed
    features_dir = '../data/features/'  # Directory containing feature CSVs
    
    X, y = load_and_extract_features(labels_csv, features_dir)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Selection: Random Forest
    rf = RandomForestClassifier(random_state=42)
    
    # Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_rf = grid_search.best_estimator_
    
    # Save the trained model and scaler
    os.makedirs('../models', exist_ok=True)
    joblib.dump(best_rf, '../models/random_forest_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    
    # Evaluation
    y_pred = best_rf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
if __name__ == "__main__":
    main()
