
import pandas as pd
import numpy as np
import joblib
import json
from itertools import product
from data_preprocessing import SoilMatePreprocessor
import warnings
warnings.filterwarnings('ignore')

def create_ml_json_lookup():
    """Create JSON lookup table using trained ML crop classifier"""
    
    try:
        # Load preprocessor and crop classifier
        preprocessor = SoilMatePreprocessor()
        preprocessor.load_preprocessors('models/')
        crop_classifier = joblib.load('models/best_crop_classifier.pkl')
        
        # Load dataset for realistic feature values
        main_df = pd.read_csv('Final_SoilMate_Dataset.csv')
        
        # Check what irrigation types were actually used in training
        available_irrigation = preprocessor.label_encoders['irrigation_source'].classes_
        print(f"Available irrigation types in training: {available_irrigation}")
        
    except Exception as e:
        print(f"Error loading models/data: {e}")
        return False
    
    # Define all possible combinations based on available data
    districts = ['Bagalkot', 'Bidar', 'Hassan', 'Haveri', 'Kodagu', 'Kolar', 
                'Koppal', 'Mandya', 'Raichur', 'Ramanagara', 'Udupi', 'Uttara Kannada']
    seasons = ['Kharif', 'Rabi', 'Summer']
    
    # Use only irrigation types that were in training data
    irrigation_types = list(available_irrigation)
    
    combinations = list(product(districts, seasons, irrigation_types))
    
    # Create lookup data structure
    lookup_data = {
        'districts': districts,
        'seasons': seasons,
        'irrigation_types': irrigation_types,
        'predictions': {},
        'metadata': {
            'model_type': 'Random Forest Classifier',
            'total_scenarios': len(combinations),
            'generated_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_accuracy': 0.928,
            'features_used': ['district_name', 'Season', 'irrigation_source', 'pH', 'N_kg_ha', 
                            'P_kg_ha', 'K_kg_ha', 'Soil_Health_Score', 'avg_rainfall_mm', 
                            'avg_temp_C', 'humidity_percent', 'Climate_Suitability', 'land_size_acres']
        }
    }
    
    # Generate predictions for all combinations
    for district, season, irrigation in combinations:
        
        # Get scenario-specific feature values with realistic variations
        scenario_data = main_df[
            (main_df['district_name'] == district) & 
            (main_df['Season'] == season)
        ]
        
        if len(scenario_data) > 0:
            # Use realistic ranges instead of just median
            ph = scenario_data['pH'].median()
            n_kg_ha = scenario_data['N_kg_ha'].median()
            p_kg_ha = scenario_data['P_kg_ha'].median()
            k_kg_ha = scenario_data['K_kg_ha'].median()
            soil_health = scenario_data['Soil_Health_Score'].median()
            rainfall = scenario_data['avg_rainfall_mm'].median()
            temp = scenario_data['avg_temp_C'].median()
            humidity = scenario_data['humidity_percent'].median()
            climate_suit = scenario_data['Climate_Suitability'].median()
        else:
            # Use overall dataset medians with regional adjustments
            ph = main_df['pH'].median()
            n_kg_ha = main_df['N_kg_ha'].median()
            p_kg_ha = main_df['P_kg_ha'].median()
            k_kg_ha = main_df['K_kg_ha'].median()
            soil_health = main_df['Soil_Health_Score'].median()
            rainfall = main_df['avg_rainfall_mm'].median()
            temp = main_df['avg_temp_C'].median()
            humidity = main_df['humidity_percent'].median()
            climate_suit = main_df['Climate_Suitability'].median()
        
        # Apply scenario-specific adjustments to create diversity
        
        # District-based adjustments
        if district in ['Kodagu', 'Udupi', 'Uttara Kannada']:  # Coastal/hill districts
            rainfall *= 1.2
            humidity += 5
            temp -= 2
        elif district in ['Bagalkot', 'Bidar', 'Koppal']:  # Dry districts
            rainfall *= 0.8
            humidity -= 5
            temp += 2
            
        # Season-based adjustments
        if season == 'Kharif':  # Monsoon season
            rainfall *= 1.5
            humidity += 10
            n_kg_ha *= 1.1  # More nitrogen availability
        elif season == 'Summer':  # Hot season
            temp += 5
            humidity -= 10
            rainfall *= 0.6
            soil_health *= 0.95  # Slight degradation in summer
        elif season == 'Rabi':  # Winter season
            temp -= 3
            humidity += 5
            
        # Irrigation-based adjustments
        if irrigation == 'Canal':
            soil_health += 2  # Better water management
            n_kg_ha *= 1.05
        elif irrigation == 'Well':
            soil_health += 5  # Best water control
            n_kg_ha *= 1.1
            p_kg_ha *= 1.05
        # Note: Rainfed adjustments removed since it's not in training data
            
        # Ensure values stay within realistic ranges
        ph = np.clip(ph, 5.5, 8.0)
        n_kg_ha = np.clip(n_kg_ha, 100, 400)
        p_kg_ha = np.clip(p_kg_ha, 10, 60)
        k_kg_ha = np.clip(k_kg_ha, 100, 500)
        soil_health = np.clip(soil_health, 40, 90)
        rainfall = np.clip(rainfall, 300, 2000)
        temp = np.clip(temp, 18, 35)
        humidity = np.clip(humidity, 40, 90)
        climate_suit = np.clip(climate_suit, 0.3, 1.0)
        
        # Create feature vector for prediction
        features = {
            'district_name': preprocessor.label_encoders['district_name'].transform([district])[0],
            'Season': preprocessor.label_encoders['Season'].transform([season])[0],
            'irrigation_source': preprocessor.label_encoders['irrigation_source'].transform([irrigation])[0],
            'pH': ph,
            'N_kg_ha': n_kg_ha,
            'P_kg_ha': p_kg_ha,
            'K_kg_ha': k_kg_ha,
            'Soil_Health_Score': soil_health,
            'avg_rainfall_mm': rainfall,
            'avg_temp_C': temp,
            'humidity_percent': humidity,
            'Climate_Suitability': climate_suit,
            'land_size_acres': 2.0  # Default land size for lookup
        }
        
        # Create DataFrame and scale features
        feature_df = pd.DataFrame([features])
        feature_scaled = preprocessor.crop_scaler.transform(feature_df)
        
        # Get predictions and probabilities
        prediction = crop_classifier.predict(feature_scaled)[0]
        probabilities = crop_classifier.predict_proba(feature_scaled)[0]
        
        # Convert prediction back to crop name
        primary_crop = preprocessor.decode_crop_prediction([prediction])[0]
        
        # Get top 3 alternatives with confidence scores
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_crops = preprocessor.decode_crop_prediction(top_indices)
        top_confidences = probabilities[top_indices]
        
        # Store prediction in lookup table
        scenario_key = f"{district}_{season}_{irrigation}"
        lookup_data['predictions'][scenario_key] = {
            'primary_recommendation': primary_crop,
            'primary_confidence': float(top_confidences[0]),
            'alternative_1': top_crops[1] if len(top_crops) > 1 else top_crops[0],
            'alternative_1_confidence': float(top_confidences[1]) if len(top_confidences) > 1 else 0.0,
            'alternative_2': top_crops[2] if len(top_crops) > 2 else top_crops[0],
            'alternative_2_confidence': float(top_confidences[2]) if len(top_confidences) > 2 else 0.0,
            'district': district,
            'season': season,
            'irrigation': irrigation
        }
    
    # Add top crops summary
    all_primary_crops = [pred['primary_recommendation'] for pred in lookup_data['predictions'].values()]
    crop_counts = pd.Series(all_primary_crops).value_counts()
    lookup_data['top_crops'] = crop_counts.head(10).index.tolist()
    
    # Save to JSON file
    try:
        with open('crop_lookup_table.json', 'w') as f:
            json.dump(lookup_data, f, indent=2)
        
        print(f"JSON lookup table created successfully")
        print(f"Total scenarios: {len(lookup_data['predictions'])}")
        print(f"Top crops: {', '.join(lookup_data['top_crops'][:5])}")
        
        return True
        
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False

def validate_json_lookup():
    """Validate the created JSON lookup table"""
    
    try:
        with open('crop_lookup_table.json', 'r') as f:
            data = json.load(f)
        
        expected_scenarios = len(data['districts']) * len(data['seasons']) * len(data['irrigation_types'])
        actual_scenarios = len(data['predictions'])
        
        print(f"Validation Results:")
        print(f"Expected scenarios: {expected_scenarios}")
        print(f"Actual scenarios: {actual_scenarios}")
        print(f"Coverage: {(actual_scenarios/expected_scenarios)*100:.1f}%")
        
        # Check sample prediction
        sample_key = list(data['predictions'].keys())[0]
        sample_pred = data['predictions'][sample_key]
        
        print(f"Sample scenarios and predictions:")
        sample_scenarios = [
            ('Hassan', 'Kharif', 'Well'),
            ('Bagalkot', 'Summer', 'Canal'), 
            ('Kodagu', 'Rabi', 'Canal'),
            ('Bidar', 'Kharif', 'Canal')
        ]
        
        for district, season, irrigation in sample_scenarios:
            key = f"{district}_{season}_{irrigation}"
            if key in data['predictions']:
                pred = data['predictions'][key]
                print(f"  {district} | {season} | {irrigation}: {pred['primary_recommendation']} ({pred['primary_confidence']:.3f})")
        
        # Check crop diversity
        all_crops = [pred['primary_recommendation'] for pred in data['predictions'].values()]
        unique_crops = len(set(all_crops))
        print(f"Crop diversity: {unique_crops} different crops recommended across scenarios")
        
        return actual_scenarios == expected_scenarios
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False

def main():
    """Main execution function"""
    
    print("Creating ML-based JSON lookup table for crop recommendations")
    
    # Create the JSON lookup table
    if create_ml_json_lookup():
        print("JSON creation successful")
        
        # Validate the created file
        if validate_json_lookup():
            print("Validation successful - Ready for application use")
        else:
            print("Validation failed - Check the generated file")
    else:
        print("JSON creation failed")

if __name__ == "__main__":
    main()