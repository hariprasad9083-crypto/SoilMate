import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SoilMate - Smart Farming Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #90EE90;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .ml-badge {
        background-color: #007bff;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    """Load all models and supporting data"""
    try:
        # Load JSON lookup table
        with open('crop_lookup_table.json', 'r') as f:
            lookup_data = json.load(f)
        return {'type': 'lookup', 'data': lookup_data}
    except FileNotFoundError:
        st.error("Crop lookup table not found! Please ensure 'crop_lookup_table.json' is available.")
        return None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

@st.cache_resource
def load_ml_economic_models():
    """Load ML economic models"""
    try:
        # Load economic models directly
        from data_preprocessing import SoilMatePreprocessor
        
        preprocessor = SoilMatePreprocessor()
        preprocessor.load_preprocessors('models/')
        
        models = {
            'cost': joblib.load('models/best_cost_model.pkl'),
            'revenue': joblib.load('models/best_revenue_model.pkl'),
            'profit': joblib.load('models/best_profit_model.pkl'),
            'preprocessor': preprocessor,
            'feature_columns': ['district_name', 'Season', 'Crop_Label', 'irrigation_source',
                              'land_size_acres', 'Soil_Health_Score', 'Climate_Suitability',
                              'Water_Requirement_mm', 'Expected_Market_Price'],
            'performance': {
                'cost_r2': 0.967,
                'revenue_r2': 0.996,
                'profit_r2': 0.993
            }
        }
        return models
    except Exception as e:
        st.sidebar.info("ML economics models not found. Using enhanced fallback analysis.")
        return None

@st.cache_data
def load_dataset():
    """Load the enhanced dataset"""
    try:
        df = pd.read_csv('Final_SoilMate_Dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'Final_SoilMate_Dataset.csv' is available.")
        return None

# Prediction functions
def predict_crop_simple(district, season, land_size, irrigation, models):
    """Get crop prediction from JSON lookup table"""
    
    if models is None or models['type'] != 'lookup':
        return None
    
    lookup_data = models['data']
    scenario_key = f"{district}_{season}_{irrigation}"
    
    if scenario_key in lookup_data['predictions']:
        prediction = lookup_data['predictions'][scenario_key]
        return {
            'primary_recommendation': prediction['primary_recommendation'],
            'alternatives': [prediction['alternative_1'], prediction['alternative_2']],
            'confidence': prediction['primary_confidence'],
            'method': 'ML Lookup Table'
        }
    
    return None

def calculate_soil_health_score(district, season, df):
    """Calculate estimated soil health score based on district and season"""
    if df is not None:
        filtered_data = df[(df['district_name'] == district) & (df['Season'] == season)]
        
        if len(filtered_data) > 0:
            avg_soil_health = filtered_data['Soil_Health_Score'].mean()
            avg_ph = filtered_data['pH'].mean()
            avg_oc = filtered_data['OC_percent'].mean()
            avg_npk = (
                filtered_data['N_kg_ha'].mean() + 
                filtered_data['P_kg_ha'].mean() + 
                filtered_data['K_kg_ha'].mean()
            ) / 3
            
            return {
                'soil_health_score': round(avg_soil_health, 1),
                'ph': round(avg_ph, 2),
                'organic_carbon': round(avg_oc, 3),
                'avg_npk': round(avg_npk, 1)
            }
    
    return {
        'soil_health_score': 75.0,
        'ph': 6.8,
        'organic_carbon': 0.45,
        'avg_npk': 250.0
    }

def predict_with_ml_models(models, farm_inputs):
    """Make predictions using trained ML models"""
    
    input_df = pd.DataFrame([farm_inputs])
    
    # Use only available features
    available_features = [col for col in models['feature_columns'] if col in input_df.columns]
    X = input_df[available_features].copy()
    
    # Fill missing values with defaults
    defaults = {
        'land_size_acres': 2.0,
        'Soil_Health_Score': 75.0,
        'Climate_Suitability': 0.75,
        'Water_Requirement_mm': 600,
        'Expected_Market_Price': 2500
    }
    
    for col, default_val in defaults.items():
        if col in X.columns:
            X[col] = X[col].fillna(default_val)
    
    # Encode categorical variables
    preprocessor = models['preprocessor']
    for col in ['district_name', 'Season', 'Crop_Label', 'irrigation_source']:
        if col in X.columns:
            try:
                encoder_key = f'economic_{col}'
                X[col] = preprocessor.label_encoders[encoder_key].transform(X[col].astype(str))
            except (KeyError, ValueError):
                X[col] = 0
    
    # Scale features
    X_scaled = preprocessor.economic_scaler.transform(X)
    
    # Make predictions
    cost_per_acre = models['cost'].predict(X_scaled)[0]
    revenue_per_acre = models['revenue'].predict(X_scaled)[0]
    profit_per_acre = models['profit'].predict(X_scaled)[0]
    
    # Scale by land size
    land_size = farm_inputs.get('land_size_acres', 2.0)
    total_cost = cost_per_acre * land_size
    total_revenue = revenue_per_acre * land_size
    total_profit = profit_per_acre * land_size
    
    roi = (total_profit / total_cost) * 100 if total_cost > 0 else 0
    
    return {
        'total_cost': total_cost,
        'total_revenue': total_revenue,
        'profit': total_profit,
        'roi_percentage': roi,
        'cost_per_acre': cost_per_acre,
        'revenue_per_acre': revenue_per_acre,
        'method': 'AI-Enhanced',
        'confidence': 0.85,
        'model_performance': models['performance']
    }

def enhanced_economic_analysis(crop, land_size, user_inputs, soil_data):
    """Enhanced economic analysis using ML models with hardcoded fallback"""
    
    ml_models = load_ml_economic_models()
    
    if ml_models is not None:
        try:
            farm_inputs = {
                'district_name': user_inputs['district'],
                'Season': user_inputs['season'],
                'Crop_Label': crop,
                'irrigation_source': user_inputs['irrigation_source'],
                'land_size_acres': land_size,
                'Soil_Health_Score': soil_data['soil_health_score'],
                'Climate_Suitability': 0.75,
                'Water_Requirement_mm': 600,
                'Expected_Market_Price': 2500
            }
            
            ml_results = predict_with_ml_models(ml_models, farm_inputs)
            
            cost_per_acre = ml_results['cost_per_acre']
            ml_results['cost_breakdown'] = {
                'seed': cost_per_acre * 0.08 * land_size,
                'fertilizer': cost_per_acre * 0.35 * land_size,
                'pesticide': cost_per_acre * 0.15 * land_size,
                'labor': cost_per_acre * 0.32 * land_size,
                'irrigation': cost_per_acre * 0.10 * land_size
            }
            
            return ml_results
            
        except Exception as e:
            st.warning(f"ML prediction failed: {e}. Using fallback analysis.")
            return fallback_economic_analysis(crop, land_size, user_inputs, soil_data)
    
    else:
        return fallback_economic_analysis(crop, land_size, user_inputs, soil_data)

def fallback_economic_analysis(crop, land_size, user_inputs, soil_data):
    """Enhanced fallback economic analysis"""
    
    base_cost_lookup = {
        'Rice': 32000, 'Wheat': 25000, 'Cotton': 44000, 'Sugarcane': 58000,
        'Soybean': 31500, 'Tomato': 67000, 'Onion': 56000, 'Groundnut': 35000,
        'Maize': 28000, 'Jowar': 26000, 'Ragi': 28000, 'Chilli': 52000,
        'Sunflower': 29000, 'Tur': 31000, 'Green Gram': 30000
    }
    
    base_revenue_lookup = {
        'Rice': {'High': 45000, 'Medium': 35000, 'Low': 25000},
        'Wheat': {'High': 40000, 'Medium': 30000, 'Low': 20000},
        'Cotton': {'High': 80000, 'Medium': 60000, 'Low': 40000},
        'Sugarcane': {'High': 120000, 'Medium': 90000, 'Low': 60000},
        'Soybean': {'High': 50000, 'Medium': 38000, 'Low': 26000},
        'Tomato': {'High': 150000, 'Medium': 100000, 'Low': 60000},
        'Onion': {'High': 120000, 'Medium': 80000, 'Low': 50000},
        'Groundnut': {'High': 55000, 'Medium': 42000, 'Low': 30000},
        'Ragi': {'High': 48000, 'Medium': 38000, 'Low': 28000},
        'Chilli': {'High': 140000, 'Medium': 95000, 'Low': 55000}
    }
    
    base_cost = base_cost_lookup.get(crop, 30000)
    
    district = user_inputs['district']
    if district in ['Hassan', 'Kodagu']:
        base_cost *= 1.05
    elif district in ['Bagalkot', 'Bidar']:
        base_cost *= 0.95
    
    irrigation = user_inputs['irrigation_source']
    if irrigation == 'Canal':
        base_cost *= 0.9
    
    soil_health = soil_data['soil_health_score']
    if soil_health < 60:
        base_cost *= 1.1
    elif soil_health > 80:
        base_cost *= 0.95
    
    cost_per_acre = base_cost
    total_cost = cost_per_acre * land_size
    
    if soil_health >= 80 and irrigation in ['Well', 'Canal']:
        yield_potential = 'High'
    elif soil_health >= 60:
        yield_potential = 'Medium'
    else:
        yield_potential = 'Low'
    
    if crop in base_revenue_lookup:
        base_revenue = base_revenue_lookup[crop][yield_potential]
    else:
        base_revenue = {'High': 50000, 'Medium': 35000, 'Low': 25000}[yield_potential]
    
    season = user_inputs['season']
    season_multiplier = {
        'Kharif': 1.0,
        'Rabi': 1.05,
        'Summer': 0.95
    }
    
    revenue_per_acre = base_revenue * season_multiplier.get(season, 1.0)
    total_revenue = revenue_per_acre * land_size
    
    profit = total_revenue - total_cost
    roi_percentage = (profit / total_cost) * 100 if total_cost > 0 else 0
    
    cost_breakdown = {
        'seed': cost_per_acre * 0.08 * land_size,
        'fertilizer': cost_per_acre * 0.35 * land_size,
        'pesticide': cost_per_acre * 0.15 * land_size,
        'labor': cost_per_acre * 0.32 * land_size,
        'irrigation': cost_per_acre * 0.10 * land_size
    }
    
    return {
        'total_cost': total_cost,
        'total_revenue': total_revenue,
        'profit': profit,
        'roi_percentage': roi_percentage,
        'cost_breakdown': cost_breakdown,
        'cost_per_acre': cost_per_acre,
        'revenue_per_acre': revenue_per_acre,
        'method': 'Enhanced Logic',
        'confidence': 0.75
    }

def get_farming_recommendations(crop, district, season, soil_health, irrigation_source):
    """Get specific farming recommendations"""
    
    recommendations = {
        'fertilizer': {
            'Rice': 'Apply 120:60:40 kg NPK per acre. Use organic manure 2-3 tons per acre.',
            'Wheat': 'Apply 100:50:30 kg NPK per acre. Add zinc sulfate 25 kg per acre.',
            'Cotton': 'Apply 150:75:75 kg NPK per acre. Use DAP and urea in split doses.',
            'Sugarcane': 'Apply 250:125:125 kg NPK per acre. Heavy organic matter required.',
            'Ragi': 'Apply 60:30:30 kg NPK per acre. Organic manure 1-2 tons per acre.',
            'Tomato': 'Apply 200:100:100 kg NPK per acre. Use calcium and magnesium supplements.',
            'Groundnut': 'Apply 25:50:75 kg NPK per acre. Gypsum application beneficial.',
            'Maize': 'Apply 120:60:40 kg NPK per acre. Split nitrogen application.',
            'Chilli': 'Apply 150:75:75 kg NPK per acre. Regular micronutrient spray.',
            'Onion': 'Apply 100:50:50 kg NPK per acre. Sulfur application recommended.'
        },
        
        'irrigation': {
            'Well': f'For {crop}, irrigate every 7-10 days. Total water requirement: 600-800mm.',
            'Canal': f'Utilize canal water efficiently. Plan irrigation based on canal schedule.'
        },
        
        'timing': {
            'Kharif': 'Plant during June-July. Harvest in October-November.',
            'Rabi': 'Plant during November-December. Harvest in March-April.',
            'Summer': 'Plant during February-March. Harvest in May-June.'
        },
        
        'best_practices': [
            f'Maintain soil pH between 6.0-7.5 for optimal {crop} growth',
            'Use certified seeds from authorized dealers',
            'Practice crop rotation to maintain soil health',
            'Monitor weather conditions and pest attacks regularly',
            'Maintain proper spacing between plants',
            'Use integrated pest management (IPM) techniques'
        ]
    }
    
    return {
        'fertilizer_advice': recommendations['fertilizer'].get(crop, 'Apply balanced NPK fertilizer as per soil test.'),
        'irrigation_advice': recommendations['irrigation'].get(irrigation_source, 'Follow standard irrigation practices.'),
        'timing_advice': recommendations['timing'].get(season, 'Follow local agricultural calendar.'),
        'best_practices': recommendations['best_practices']
    }

# Main Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">SoilMate - Smart Farming Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your AI-powered companion for better farming decisions</p>', unsafe_allow_html=True)
    
    # Load models and data
    models = load_models()
    df = load_dataset()
    
    if models is None:
        st.error("No trained models found! Please ensure model files are in the directory.")
        return
    
    # Sidebar for inputs
    st.sidebar.markdown("## Farm Information")
    st.sidebar.markdown("Please provide your farm details:")
    
    # Get available options from JSON
    if models and models['type'] == 'lookup':
        districts = models['data']['districts']
        seasons = models['data']['seasons']
        irrigation_types = models['data']['irrigation_types']
    else:
        districts = ['Bagalkot', 'Hassan', 'Kodagu', 'Mandya']
        seasons = ['Kharif', 'Rabi', 'Summer']
        irrigation_types = ['Canal', 'Well']
    
    # Input 1: Location (District)
    district = st.sidebar.selectbox(
        "Which district is your farm located in?",
        districts,
        help="Select your farm's district location"
    )
    
    # Input 2: Farming Season
    season_options = {
        "Monsoon Season (Jun-Oct)": "Kharif",
        "Winter Season (Nov-Feb)": "Rabi", 
        "Summer Season (Mar-May)": "Summer"
    }
    season_display = st.sidebar.selectbox(
        "When are you planning to plant?",
        list(season_options.keys()),
        help="Choose the planting season"
    )
    season = season_options[season_display]
    
    # Input 3: Farm Size
    land_size = st.sidebar.slider(
        "How much land do you want to cultivate? (acres)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Specify your farm size in acres"
    )
    
    # Input 4: Water Source
    irrigation_display_options = {
        "Well/Borewell": "Well",
        "Canal/River": "Canal"
    }
    
    # Filter based on available irrigation types
    available_irrigation_display = {}
    for display, value in irrigation_display_options.items():
        if value in irrigation_types:
            available_irrigation_display[display] = value
    
    irrigation_display = st.sidebar.selectbox(
        "What is your main source of water?",
        list(available_irrigation_display.keys()),
        help="Select your primary water source"
    )
    irrigation_source = available_irrigation_display[irrigation_display]
    
    # Get Recommendations Button
    if st.sidebar.button("Get Recommendations", type="primary"):
        st.session_state.show_recommendations = True
        st.session_state.user_inputs = {
            'district': district,
            'season': season,
            'land_size': land_size,
            'irrigation_source': irrigation_source
        }
    
    # Display recommendations if button clicked
    if hasattr(st.session_state, 'show_recommendations') and st.session_state.show_recommendations:
        display_recommendations(st.session_state.user_inputs, models, df)

def display_recommendations(inputs, models, df):
    """Display comprehensive recommendations"""
    
    # Get predictions
    crop_prediction = predict_crop_simple(
        inputs['district'], inputs['season'], inputs['land_size'],
        inputs['irrigation_source'], models
    )
    
    if crop_prediction is None:
        st.error("No recommendation available for this combination.")
        return
    
    soil_data = calculate_soil_health_score(inputs['district'], inputs['season'], df)
    
    # Enhanced economic analysis (ML or fallback)
    economics = enhanced_economic_analysis(
        crop_prediction['primary_recommendation'], 
        inputs['land_size'], 
        inputs,
        soil_data
    )
    
    recommendations = get_farming_recommendations(
        crop_prediction['primary_recommendation'],
        inputs['district'],
        inputs['season'],
        soil_data['soil_health_score'],
        inputs['irrigation_source']
    )
    
    # Create tabs for organized display
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Crop Recommendations",
        "Soil Analysis", 
        "Economic Analysis",
        "Farming Guide",
        "Comparative Analysis"
    ])
    
    with tab1:
        display_crop_recommendations(crop_prediction, inputs)
    
    with tab2:
        display_soil_analysis(soil_data, inputs)
    
    with tab3:
        display_enhanced_economic_analysis(economics, inputs)
    
    with tab4:
        display_farming_guide(recommendations, inputs)
    
    with tab5:
        display_comparative_analysis(inputs, df, crop_prediction['primary_recommendation'])

def display_crop_recommendations(prediction, inputs):
    """Display crop recommendation section"""
    st.markdown('<h2 class="sub-header">Crop Recommendations</h2>', unsafe_allow_html=True)
    
    # Primary recommendation
    st.markdown(f"""
    <div class="recommendation-box">
        <h3>Primary Recommendation</h3>
        <h2 style="color: #2E8B57; margin: 0;">{prediction['primary_recommendation']}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Alternative recommendations
    col1, col2 = st.columns(2)
    
    for i, crop in enumerate(prediction['alternatives'][:2]):
        with [col1, col2][i]:
            st.metric(
                label=f"Alternative {i+1}",
                value=crop
            )
    
    # Reasoning
    st.markdown("### Why This Recommendation?")
    
    reasons = []
    if inputs['season'] == 'Kharif':
        reasons.append("Monsoon season provides adequate water for growth")
    elif inputs['season'] == 'Rabi':
        reasons.append("Winter season offers optimal temperature conditions")
    else:
        reasons.append("Summer cultivation with proper irrigation management")
    
    if inputs['irrigation_source'] == 'Canal':
        reasons.append("Canal irrigation provides reliable water supply")
    elif inputs['irrigation_source'] == 'Well':
        reasons.append("Well irrigation offers flexible water management")
    
    reasons.append(f"{inputs['district']} district has favorable conditions for this crop")
    
    if inputs['land_size'] >= 5.0:
        reasons.append("Large farm size suitable for commercial cultivation")
    elif inputs['land_size'] <= 1.0:
        reasons.append("Small farm optimized for high-value crops")
    
    for reason in reasons:
        st.write(f"• {reason}")

def display_soil_analysis(soil_data, inputs):
    """Display soil analysis section"""
    st.markdown('<h2 class="sub-header">Soil Health Analysis</h2>', unsafe_allow_html=True)
    
    # Soil health metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Soil Health Score", f"{soil_data['soil_health_score']}/100")
    
    with col2:
        st.metric("pH Level", soil_data['ph'])
    
    with col3:
        st.metric("Organic Carbon", f"{soil_data['organic_carbon']}%")
    
    with col4:
        st.metric("Avg NPK", f"{soil_data['avg_npk']:.0f} kg/ha")
    
    # Soil health gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = soil_data['soil_health_score'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Soil Health Score"},
        delta = {'reference': 75},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Soil recommendations
    if soil_data['soil_health_score'] >= 80:
        st.markdown("""
        <div class="success-box">
            <strong>Excellent Soil Health!</strong><br>
            Your soil is in great condition. Maintain current practices and consider organic farming methods.
        </div>
        """, unsafe_allow_html=True)
    elif soil_data['soil_health_score'] >= 60:
        st.markdown("""
        <div class="warning-box">
            <strong>Moderate Soil Health</strong><br>
            Consider adding organic matter and balanced fertilization to improve soil condition.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>Soil Needs Attention</strong><br>
            Immediate soil improvement measures recommended. Consider soil testing and organic amendments.
        </div>
        """, unsafe_allow_html=True)

def display_enhanced_economic_analysis(economics, inputs):
    """Display enhanced economic analysis section with market intelligence"""
    
    st.markdown('<h2 class="sub-header">Economic Analysis & Market Intelligence</h2>', unsafe_allow_html=True)
    
    # Key metrics with confidence indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Investment", f"₹{economics['total_cost']:,.0f}")
    
    with col2:
        st.metric("Expected Revenue", f"₹{economics['total_revenue']:,.0f}")
    
    with col3:
        profit_color = "normal" if economics['profit'] >= 0 else "inverse"
        st.metric("Projected Profit", f"₹{economics['profit']:,.0f}", delta_color=profit_color)
    
    with col4:
        st.metric("ROI", f"{economics['roi_percentage']:.1f}%")
    
    # Cost breakdown chart
    cost_breakdown = economics['cost_breakdown']
    
    fig = px.pie(
        values=list(cost_breakdown.values()),
        names=list(cost_breakdown.keys()),
        title="Cost Breakdown Analysis",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Market intelligence section
    st.markdown("### Market Intelligence")
    
    # Simulated market data for demonstration
    market_col1, market_col2 = st.columns(2)
    
    with market_col1:
        # Price trend simulation
        dates = pd.date_range('2024-01-01', periods=12, freq='M')
        prices = np.random.normal(2500, 200, 12).cumsum() - np.random.normal(2500, 200, 12).cumsum()[0] + 2500
        
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines+markers',
            name='Market Price',
            line=dict(color='#2E8B57', width=3)
        ))
        
        price_fig.update_layout(
            title="Market Price Trends",
            xaxis_title="Month",
            yaxis_title="Price (₹/quintal)",
            height=300,
            template="plotly_white"
        )
        
        st.plotly_chart(price_fig, use_container_width=True)
    
    with market_col2:
        # Market insights
        current_price = prices[-1]
        price_change = ((prices[-1] - prices[-2]) / prices[-2]) * 100
        
        st.metric("Current Market Price", f"₹{current_price:.0f}/quintal", 
                 delta=f"{price_change:+.1f}%")
        
        st.markdown("""
        **Market Insights:**
        - Price volatility: Moderate
        - Seasonal demand: Increasing
        """)
    
    # Enhanced profitability analysis
    st.markdown("### Profitability Analysis")
    
    if economics['roi_percentage'] >= 25:
        st.markdown("""
        <div class="success-box">
            <strong>Highly Profitable Investment!</strong><br>
            Expected ROI of {:.1f}% indicates excellent returns. This is a great investment opportunity.
        </div>
        """.format(economics['roi_percentage']), unsafe_allow_html=True)
    elif economics['roi_percentage'] >= 15:
        st.markdown("""
        <div class="success-box">
            <strong>Good Investment Returns</strong><br>
            Expected ROI of {:.1f}% shows solid profitability. Consider this opportunity.
        </div>
        """.format(economics['roi_percentage']), unsafe_allow_html=True)
    elif economics['roi_percentage'] >= 5:
        st.markdown("""
        <div class="warning-box">
            <strong>Moderate Returns</strong><br>
            Expected ROI of {:.1f}% is acceptable but consider cost optimization strategies.
        </div>
        """.format(economics['roi_percentage']), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>Low Profitability</strong><br>
            Expected ROI of {:.1f}% is below target. Review crop selection or reduce input costs.
        </div>
        """.format(economics['roi_percentage']), unsafe_allow_html=True)

def display_farming_guide(recommendations, inputs):
    """Display farming guide section - Prescriptive Analytics"""
    st.markdown('<h2 class="sub-header">Comprehensive Farming Guide</h2>', unsafe_allow_html=True)
    
    # Fertilizer recommendations
    st.markdown("### Fertilizer Recommendations")
    st.markdown(f"""
    <div class="metric-container">
        {recommendations['fertilizer_advice']}
    </div>
    """, unsafe_allow_html=True)
    
    # Irrigation guide
    st.markdown("### Irrigation Management")
    st.markdown(f"""
    <div class="metric-container">
        {recommendations['irrigation_advice']}
    </div>
    """, unsafe_allow_html=True)
    
    # Timing recommendations
    st.markdown("### Planting & Harvesting Timeline")
    st.markdown(f"""
    <div class="metric-container">
        {recommendations['timing_advice']}
    </div>
    """, unsafe_allow_html=True)
    
    # Best practices
    st.markdown("### Best Practices")
    for practice in recommendations['best_practices']:
        st.write(f"✅ {practice}")
    
    # Weather considerations
    st.markdown("### Weather Considerations")
    
    weather_tips = {
        'Kharif': [
            "Monitor monsoon patterns and adjust planting dates",
            "Ensure proper drainage to prevent waterlogging",
            "Watch for pest outbreaks during humid conditions"
        ],
        'Rabi': [
            "Take advantage of cool, dry weather for better growth",
            "Plan for supplemental irrigation during dry spells",
            "Protect crops from unexpected frost"
        ],
        'Summer': [
            "Ensure adequate water supply throughout the season",
            "Use mulching to conserve soil moisture",
            "Plan for higher irrigation frequency"
        ]
    }
    
    for tip in weather_tips.get(inputs['season'], []):
        st.write(f"• {tip}")
def display_comparative_analysis(inputs, df, recommended_crop):
    """Display enhanced comparative analysis section"""
    st.markdown('<h2 class="sub-header">Comparative Analysis</h2>', unsafe_allow_html=True)
    
    if df is not None:
        # 1. Crop Profitability Ranking
        st.markdown("### Top 10 Most Profitable Crops")
        
        profit_columns = ['Crop_Profitability_Index', 'ROI_Percentage', 'Expected_Revenue']
        available_profit_columns = [col for col in profit_columns if col in df.columns]
        
        if available_profit_columns and 'Crop_Label' in df.columns:
            crop_performance = df.groupby('Crop_Label')[available_profit_columns].mean().round(2)
            
            # Sort by profitability index or ROI
            sort_column = 'Crop_Profitability_Index' if 'Crop_Profitability_Index' in crop_performance.columns else 'ROI_Percentage'
            if sort_column in crop_performance.columns:
                top_crops = crop_performance.sort_values(sort_column, ascending=False).head(10)
                
                # Create horizontal bar chart
                crop_fig = px.bar(
                    top_crops.reset_index(),
                    x=sort_column,
                    y='Crop_Label',
                    title=f"Top 10 Most Profitable Crops (Based on {sort_column.replace('_', ' ')})",
                    labels={sort_column: sort_column.replace('_', ' '), 'Crop_Label': 'Crop'},
                    color=sort_column,
                    color_continuous_scale='Viridis',
                    orientation='h'
                )
                
                crop_fig.update_layout(height=500)
                st.plotly_chart(crop_fig, use_container_width=True)
                
                # Highlight recommended crop position
                if recommended_crop in top_crops.index:
                    crop_rank = list(top_crops.index).index(recommended_crop) + 1
                    crop_score = top_crops.loc[recommended_crop, sort_column]
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>Your Recommended Crop ({recommended_crop}):</strong><br>
                        Ranks #{crop_rank} out of top 10 most profitable crops with {sort_column.replace('_', ' ')}: {crop_score:.2f}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                        <strong>Your Recommended Crop ({recommended_crop}):</strong><br>
                        Not in top 10 most profitable crops overall, but may be optimal for your specific conditions.
                    </div>
                    """, unsafe_allow_html=True)
        
        # 2. District-Specific Crop Analysis
        st.markdown(f"### Crop Analysis for {inputs['district']} District")
        
        district_data = df[df['district_name'] == inputs['district']]
        if len(district_data) > 0 and 'Crop_Label' in district_data.columns:
            
            top_crops_district = district_data['Crop_Label'].value_counts().head(8)
            
            if len(top_crops_district) > 0:
                crop_comparison_data = []
                
                for crop in top_crops_district.index:
                    crop_data = district_data[district_data['Crop_Label'] == crop]
                    if len(crop_data) > 0:
                        avg_soil_health = crop_data['Soil_Health_Score'].mean()
                        avg_roi = crop_data['ROI_Percentage'].mean() if 'ROI_Percentage' in crop_data.columns else 0
                        avg_profit = crop_data['Profit_Per_Acre'].mean() if 'Profit_Per_Acre' in crop_data.columns else 0
                        
                        # Use actual data if available, otherwise fallback
                        if avg_roi == 0 or avg_profit == 0:
                            temp_economics = fallback_economic_analysis(crop, 1.0, inputs, {
                                'soil_health_score': avg_soil_health,
                                'ph': 6.8,
                                'organic_carbon': 0.45
                            })
                            avg_roi = temp_economics['roi_percentage']
                            avg_profit = temp_economics['revenue_per_acre'] - temp_economics['cost_per_acre']
                        
                        crop_comparison_data.append({
                            'Crop': crop,
                            'Soil_Health': avg_soil_health,
                            'ROI': avg_roi,
                            'Profit_Per_Acre': avg_profit,
                            'Popularity': top_crops_district[crop],
                            'Recommended': 'Recommended' if crop == recommended_crop else ''
                        })
                
                if crop_comparison_data:
                    comparison_df = pd.DataFrame(crop_comparison_data)
                    
                    # Risk vs Reward scatter plot
                    scatter_fig = px.scatter(
                        comparison_df,
                        x='ROI',
                        y='Profit_Per_Acre',
                        size='Popularity',
                        color='Soil_Health',
                        hover_data=['Crop', 'Recommended'],
                        title=f"Risk vs Reward Analysis - {inputs['district']} District",
                        labels={'ROI': 'ROI (%)', 'Profit_Per_Acre': 'Profit per Acre (₹)'},
                        color_continuous_scale='Viridis'
                    )
                    
                    # Add quadrant lines
                    avg_roi = comparison_df['ROI'].mean()
                    avg_profit = comparison_df['Profit_Per_Acre'].mean()
                    
                    scatter_fig.add_hline(y=avg_profit, line_dash="dash", line_color="red", 
                                        annotation_text="Avg Profit")
                    scatter_fig.add_vline(x=avg_roi, line_dash="dash", line_color="red", 
                                        annotation_text="Avg ROI")
                    
                    scatter_fig.update_layout(height=500)
                    st.plotly_chart(scatter_fig, use_container_width=True)
                    
                    st.markdown("**Analysis Guide:** Top-right quadrant = High ROI + High Profit (Best crops)")
            
    else:
        st.info("Dataset not available for comparative analysis.")

# Main execution
if __name__ == "__main__":
    main()