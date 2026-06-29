import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

class SoilMatePreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.crop_scaler = StandardScaler()
        self.economic_scaler = StandardScaler()
        self.feature_columns = []
        self.target_encoders = {}
        
    def load_data(self, main_dataset_path, price_dataset_path):
        """Load and merge datasets"""
        self.main_df = pd.read_csv(main_dataset_path)
        self.price_df = pd.read_csv(price_dataset_path)
        return self.main_df, self.price_df
    
    def prepare_crop_classification_data(self):
        """Prepare data for crop recommendation model"""
        
        # Select key features for crop classification
        feature_cols = [
            'district_name', 'Season', 'irrigation_source',
            'pH', 'N_kg_ha', 'P_kg_ha', 'K_kg_ha', 'Soil_Health_Score',
            'avg_rainfall_mm', 'avg_temp_C', 'humidity_percent', 
            'Climate_Suitability', 'land_size_acres'
        ]
        
        # Prepare features and target
        X = self.main_df[feature_cols].copy()
        y = self.main_df['Crop_Label'].copy()
        
        # Handle categorical features
        categorical_cols = ['district_name', 'Season', 'irrigation_source']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Encode target variable
        if 'crop_target' not in self.target_encoders:
            self.target_encoders['crop_target'] = LabelEncoder()
            y_encoded = self.target_encoders['crop_target'].fit_transform(y)
        else:
            y_encoded = self.target_encoders['crop_target'].transform(y)
        
        # Store feature names
        self.feature_columns = X.columns.tolist()
        
        return X, y_encoded
    
    def prepare_economic_data(self, target_type='cost'):
        """Prepare data for economic models"""
        
        # Select key features for economic models
        feature_cols = [
            'district_name', 'Season', 'Crop_Label', 'irrigation_source',
            'land_size_acres', 'Soil_Health_Score', 'Climate_Suitability',
            'Water_Requirement_mm', 'Expected_Market_Price'
        ]
        
        X = self.main_df[feature_cols].copy()
        
        # Select target based on type
        target_mapping = {
            'cost': 'Total_Production_Cost',
            'revenue': 'Expected_Revenue', 
            'profit': 'Profit_Per_Acre'
        }
        
        y = self.main_df[target_mapping[target_type]].copy()
        
        # Handle categorical features
        categorical_cols = ['district_name', 'Season', 'Crop_Label', 'irrigation_source']
        
        for col in categorical_cols:
            encoder_key = f'economic_{col}'
            if encoder_key not in self.label_encoders:
                self.label_encoders[encoder_key] = LabelEncoder()
                X[col] = self.label_encoders[encoder_key].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[encoder_key].transform(X[col])
        
        return X, y
    
    def scale_features(self, X_train, X_test=None, fit_scaler=True, scaler_type='crop'):
        """Scale numerical features"""
        scaler = self.crop_scaler if scaler_type == 'crop' else self.economic_scaler
        
        if fit_scaler:
            X_train_scaled = scaler.fit_transform(X_train)
        else:
            X_train_scaled = scaler.transform(X_train)
            
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def create_train_test_split(self, X, y, test_size=0.2, random_state=42):
        """Create stratified train-test split"""
        return train_test_split(X, y, test_size=test_size, 
                               random_state=random_state, stratify=y)
    
    def save_preprocessors(self, save_dir='models/'):
        """Save all preprocessors"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.label_encoders, f'{save_dir}label_encoders.pkl')
        joblib.dump(self.crop_scaler, f'{save_dir}crop_scaler.pkl')
        joblib.dump(self.economic_scaler, f'{save_dir}economic_scaler.pkl')
        joblib.dump(self.target_encoders, f'{save_dir}target_encoders.pkl')
        joblib.dump(self.feature_columns, f'{save_dir}feature_columns.pkl')
    
    def load_preprocessors(self, save_dir='models/'):
        """Load all preprocessors"""
        self.label_encoders = joblib.load(f'{save_dir}label_encoders.pkl')
        self.crop_scaler = joblib.load(f'{save_dir}crop_scaler.pkl')
        self.economic_scaler = joblib.load(f'{save_dir}economic_scaler.pkl')
        self.target_encoders = joblib.load(f'{save_dir}target_encoders.pkl')
        self.feature_columns = joblib.load(f'{save_dir}feature_columns.pkl')
    
    def get_crop_classes(self):
        """Get crop class names"""
        if 'crop_target' in self.target_encoders:
            return self.target_encoders['crop_target'].classes_
        return None
    
    def decode_crop_prediction(self, encoded_prediction):
        """Convert encoded prediction back to crop name"""
        if 'crop_target' in self.target_encoders:
            return self.target_encoders['crop_target'].inverse_transform(encoded_prediction)
        return encoded_prediction

def main():
    """Test the preprocessing pipeline"""
    
    # Initialize preprocessor
    preprocessor = SoilMatePreprocessor()
    
    # Load data
    main_df, price_df = preprocessor.load_data(
        'Final_SoilMate_Dataset.csv', 
        'Final_SoilMate_Crop_Prices.csv'
    )
    
    print(f"Loaded datasets: Main {main_df.shape}, Price {price_df.shape}")
    
    # Prepare crop classification data
    X_crop, y_crop = preprocessor.prepare_crop_classification_data()
    print(f"Crop classification data: {X_crop.shape}, {len(np.unique(y_crop))} classes")
    
    # Create train-test split
    X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(X_crop, y_crop)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test, scaler_type='crop')
    print(f"Features scaled successfully")
    
    # Test economic data preparation
    for target_type in ['cost', 'revenue', 'profit']:
        X_econ, y_econ = preprocessor.prepare_economic_data(target_type)
        print(f"Economic {target_type} data: {X_econ.shape}")
        
        # Scale economic features (fit scaler on first target)
        if target_type == 'cost':
            X_econ_scaled = preprocessor.scale_features(X_econ, scaler_type='economic')
        else:
            X_econ_scaled = preprocessor.scale_features(X_econ, fit_scaler=False, scaler_type='economic')
    
    # Save preprocessors
    preprocessor.save_preprocessors()
    print("Preprocessors saved successfully")
    
    return preprocessor, X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    preprocessor, X_train, X_test, y_train, y_test = main()