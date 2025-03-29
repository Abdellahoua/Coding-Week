import streamlit as st
import pickle
import numpy as np
import xgboost as xgb
import requests
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import shap

# Set page configuration
st.set_page_config(
    page_title="Obesity Risk Prediction",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st._config.set_option("theme.base", "light")

# Apply enhanced custom CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background-color: #f5f7fa;
    }
    
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    
    .main-subheader {
        font-size: 1.1rem;
        font-weight: 400;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .subheader {
        font-size: 1.6rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1.2rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #334155;
        margin: 1.2rem 0 0.8rem 0;
        padding-bottom: 0.2rem;
    }
    
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #f1f5f9;
        transition: all 0.2s ease;
    }
    
    .card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border-color: #e2e8f0;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    
    .metric-item {
        text-align: center;
        padding: 1rem;
        min-width: 140px;
        border-radius: 8px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease;
    }
    
    .metric-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        line-height: 1.1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748b;
    }
    
    .prediction-result {
        padding: 24px;
        border-radius: 10px;
        text-align: center;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .prediction-result h2 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .prediction-result h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .tab-content {
        padding: 1.2rem;
        background-color: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 6px 10px rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    .explanation-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
    }
    
    .feature-impact {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        padding: 8px 12px;
        border-radius: 6px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    
    .feature-name {
        font-weight: 600;
        flex: 1;
    }
    
    .feature-value {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        background-color: #f1f5f9;
        margin: 0 8px;
    }
    
    .impact-indicator {
        display: flex;
        align-items: center;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .impact-positive {
        color: #ef4444;
    }
    
    .impact-negative {
        color: #10b981;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        border: 1px solid #e2e8f0;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-color: #e2e8f0;
        border-bottom: none;
        font-weight: 600;
    }
    
    .placeholder-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        color: #94a3b8;
        background-color: #f8fafc;
        border-radius: 12px;
        border: 2px dashed #e2e8f0;
    }
    
    .placeholder-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.4;
    }
    
    .placeholder-text {
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Form styling */
    div[data-baseweb="select"] {
        border-radius: 8px;
    }
    
    div[data-baseweb="select"] > div {
        background-color: white;
        border-radius: 8px;
        border-color: #e2e8f0;
    }
    
    div[data-baseweb="base-input"] {
        border-radius: 8px;
    }
    
    div[data-baseweb="base-input"] > div {
        background-color: white;
        border-radius: 8px;
        border-color: #e2e8f0;
    }
    
    /* Custom section divider */
    .section-divider {
        display: flex;
        align-items: center;
        margin: 1.5rem 0;
    }
    
    .section-divider-line {
        flex: 1;
        height: 1px;
        background-color: #e2e8f0;
    }
    
    .section-divider-text {
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        padding: 0 12px;
    }
    
    /* SHAP plots styling */
    .shap-plot-container {
        background-color: white;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
    }
    
    .shap-plot-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.5rem;
    }
    
    .shap-plot-description {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 1rem;
    }
    
    /* Additional visualization enhancements */
    .category-label {
        font-size: 1rem;
        font-weight: 600;
        text-align: center;
        padding: 8px;
        border-radius: 6px;
        margin-top: 0.5rem;
    }
    .card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 10px rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

def load_css():
    st.markdown("""
    <style>
    /* Modern, professional styling for the Health Profile tab */
    .subheader {
        color: #1e40af;
        font-weight: 600;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #dbeafe;
        padding-bottom: 0.5rem;
    }
    
    .card {
        background-color: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
    }
    
    .section-title {
        color: #334155;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .section-title::before {
        content: "";
        display: inline-block;
        width: 0.5rem;
        height: 1.25rem;
        background-color: #3b82f6;
        margin-right: 0.5rem;
        border-radius: 1rem;
    }
    
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        justify-content: space-between;
    }
    
    .metric-item {
        flex: 1 1 120px;
        background-color: #f8fafc;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .metric-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .metric-item::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        height: 4px;
        width: 100%;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .metric-info {
        position: absolute;
        top: 0.25rem;
        right: 0.25rem;
        color: #94a3b8;
        cursor: help;
    }
    
    .placeholder-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        background-color: #f1f5f9;
        border-radius: 0.75rem;
        border: 2px dashed #cbd5e1;
    }
    
    .placeholder-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: #64748b;
    }
    
    .placeholder-text {
        color: #64748b;
        font-size: 1.125rem;
        font-weight: 500;
        text-align: center;
    }
    
    .health-score-container {
        position: relative;
        width: 150px;
        height: 150px;
        margin: 0 auto 1.5rem auto;
    }
    
    .recommendation-item {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8fafc;
        margin-bottom: 0.75rem;
        border-left: 4px solid #3b82f6;
        transition: all 0.2s ease;
        display: flex;
        align-items: flex-start;
    }
    
    .recommendation-item:hover {
        background-color: #f0f9ff;
        transform: translateX(5px);
    }
    
    .recommendation-number {
        background-color: #3b82f6;
        color: white;
        width: 1.5rem;
        height: 1.5rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.75rem;
        margin-right: 0.75rem;
        flex-shrink: 0;
    }
    
    .recommendation-text {
        font-weight: 500;
        color: #334155;
        flex-grow: 1;
    }
    
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #334155;
        color: white;
        text-align: center;
        padding: 0.5rem;
        border-radius: 0.25rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.75rem;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.75rem;
        margin-right: 0.5rem;
    }
    
    .badge-success {
        background-color: #dcfce7;
        color: #15803d;
    }
    
    .badge-warning {
        background-color: #fef9c3;
        color: #a16207;
    }
    
    .badge-danger {
        background-color: #fee2e2;
        color: #b91c1c;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .metric-container {
            flex-direction: column;
        }
        
        .metric-item {
            flex: 1 1 100%;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# Load model function
def load_model():
    try:
        # Step 1: Define the URL to the raw model file in the GitHub repository
        model_url = "https://github.com/YassineLahniche/Coding-Week/raw/main/model/xgb_baseline.pkl"

        # Step 2: Download the model file
        response = requests.get(model_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Step 3: Load the model using pickle
            model = pickle.loads(response.content)
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def calculate_bmi(weight, height):
    height_m = height / 100  # Convert cm to meters
    return round(weight / (height_m ** 2), 2)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "#3b82f6"  # Blue
    elif 18.5 <= bmi < 25:
        return "Normal weight", "#10b981"  # Green
    elif 25 <= bmi < 30:
        return "Overweight", "#f59e0b"  # Amber
    elif 30 <= bmi < 35:
        return "Obesity Class I", "#f97316"  # Orange
    elif 35 <= bmi < 40:
        return "Obesity Class II", "#ef4444"  # Red
    else:
        return "Obesity Class III", "#b91c1c"  # Dark red

def predict_obesity_level(model, input_data):
    obesity_categories = {
        0: 'Insufficient Weight',
        1: 'Normal Weight',
        2: 'Obesity Type I',
        3: 'Obesity Type II',
        4: 'Obesity Type III',
        5: 'Overweight Level I',
        6: 'Overweight Level II'
    }
    
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    category = obesity_categories.get(int(prediction))
    
    return prediction, category

def get_category_color(category):
    colors = {
        'Insufficient Weight': '#3b82f6',  # Blue
        'Normal Weight': '#10b981',        # Green
        'Overweight Level I': '#f59e0b',   # Amber
        'Overweight Level II': '#f97316',  # Orange
        'Obesity Type I': '#ef4444',       # Red
        'Obesity Type II': '#dc2626',      # Darker red
        'Obesity Type III': '#b91c1c'      # Darkest red
    }
    return colors.get(category, "#94a3b8")  # Default slate gray

def create_bmi_gauge(bmi):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = bmi,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "BMI", 'font': {'size': 24, 'family': 'Poppins, sans-serif', 'color': '#1e293b'}},
        gauge = {
            'axis': {'range': [None, 45], 'tickwidth': 1, 'tickcolor': "#64748b", 'tickfont': {'family': 'Poppins, sans-serif'}},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 18.5], 'color': '#3b82f6'},   # Blue
                {'range': [18.5, 25], 'color': '#10b981'},  # Green
                {'range': [25, 30], 'color': '#f59e0b'},    # Amber
                {'range': [30, 35], 'color': '#f97316'},    # Orange
                {'range': [35, 40], 'color': '#ef4444'},    # Red
                {'range': [40, 45], 'color': '#b91c1c'}     # Dark red
            ],
            'threshold': {
                'line': {'color': "#1e293b", 'width': 4},
                'thickness': 0.75,
                'value': bmi
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#1e293b", 'family': "Poppins, sans-serif"}
    )
    return fig

def create_category_chart(category):
    categories = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 
                 'Overweight Level II', 'Obesity Type I', 'Obesity Type II', 'Obesity Type III']
    
    # Create a horizontal bar chart
    fig = go.Figure()
    
    # Add colored bars for each category
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#f97316', '#ef4444', '#dc2626', '#b91c1c']
    
    for i, cat in enumerate(categories):
        opacity = 0.3
        if cat == category:
            opacity = 1.0
        
        fig.add_trace(go.Bar(
            y=[cat],
            x=[1],
            orientation='h',
            marker=dict(color=colors[i], opacity=opacity),
            showlegend=False,
            hoverinfo='none'
        ))
    
    # Add a marker for the predicted category
    category_index = categories.index(category)
    
    fig.add_annotation(
        x=0.5,
        y=category_index,
        text="✓",
        showarrow=False,
        font=dict(size=24, color="white")
    )
    
    fig.update_layout(
        height=300,
        barmode='stack',
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", tickfont={'family': "Poppins, sans-serif", 'color': '#1e293b'}),
        font={'color': "#1e293b", 'family': "Poppins, sans-serif"}
    )
    
    return fig

def create_shap_explainer(model):
    explainer = shap.TreeExplainer(model)
    return explainer

# Calculate SHAP values for a specific instance
def calculate_shap_values(explainer, instance):
    # Handle both single and multi-class outputs
    shap_values = explainer.shap_values(instance)
    return shap_values

# Create SHAP force plot with improved styling
def create_shap_force_plot(explainer, shap_values, instance, feature_names):
    plt.figure(figsize=(12, 4))  # Slightly larger figure
    
    # Handle different types of expected values
    if hasattr(explainer, 'expected_value'):
        if isinstance(explainer.expected_value, list):
            expected_value = explainer.expected_value[0]
        else:
            expected_value = explainer.expected_value
    else:
        expected_value = 0
    
    # Create force plot with improved visualization
    shap_plot = shap.force_plot(
        expected_value, 
        shap_values, 
        instance, 
        feature_names=feature_names,
        link="logit"  # Better for classification problems
    )
    return shap_plot

# Create improved SHAP waterfall plot
def create_shap_waterfall_plot(explainer, shap_values, instance, feature_names):
    plt.figure(figsize=(10, 6))
    
    # Get the expected value (base value)
    if hasattr(explainer, 'expected_value'):
        if isinstance(explainer.expected_value, list):
            expected_value = explainer.expected_value[0]
        else:
            expected_value = explainer.expected_value
    else:
        expected_value = 0
    
    # Use a try-except block to handle different SHAP library versions
    try:
        # New SHAP version
        shap.plots.waterfall(shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=instance,
            feature_names=feature_names
        ), show=False)
    except:
        # Legacy version fallback
        shap.plots._waterfall.waterfall_legacy(
            expected_value, 
            shap_values, 
            feature_names=feature_names, 
            show=False
        )
    
    plt.title("Feature Impact on Obesity Risk Prediction", fontsize=14)
    plt.tight_layout()
    return plt

# Create improved SHAP bar plot for feature importance
def create_shap_bar_plot(explainer, input_array, feature_names):
    """Generate a SHAP feature importance bar plot with Streamlit-compatible formatting."""
    
    # Ensure SHAP values are computed correctly
    shap_values = explainer.shap_values(input_array)
    
    # Handle classification models (multi-class SHAP output)
    if isinstance(shap_values, list):  
        shap_values = np.mean(shap_values, axis=0)  # Average across classes if needed
    
    # Create SHAP summary bar plot
    plt.figure(figsize=(10, 6))  # Adjust size for better readability
    shap.summary_plot(
        shap_values, 
        input_array, 
        feature_names=feature_names, 
        plot_type="bar", 
        show=False  # Prevent automatic showing in Jupyter
    )
    
    # Enhance visualization
    plt.title("Feature Importance in Obesity Prediction", fontsize=14, fontweight="bold")
    plt.xlabel("Mean SHAP Value (Feature Importance)", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    
    # Display the plot correctly in Streamlit
    st.pyplot(plt)

# Create improved SHAP decision plot
def create_shap_decision_plot(explainer, shap_values, instance, feature_names):
    plt.figure(figsize=(12, 10))  # Larger figure for better readability
    
    # Get the expected value (base value)
    if hasattr(explainer, 'expected_value'):
        if isinstance(explainer.expected_value, list):
            expected_value = explainer.expected_value[0]
        else:
            expected_value = explainer.expected_value
    else:
        expected_value = 0
    
    # Create improved decision plot
    shap.decision_plot(
        expected_value, 
        shap_values, 
        feature_names=feature_names, 
        show=False,
        feature_display_range=slice(-1, -16, -1)  # Show most important features at top
    )
    
    plt.title("Decision Path for Obesity Risk Prediction", fontsize=14)
    plt.tight_layout()
    return plt

# Get feature names (unchanged)
def get_feature_names():
    return [
        "Gender", "Age", "Height", "Weight", "Family_History", "High_Caloric_Food",
        "Vegetable_Consumption", "Main_Meals", "Snacking", "Smoking",
        "Water_Intake", "Calorie_Monitoring", "Physical_Activity",
        "Screen_Time", "Alcohol", "Transportation"
    ]

# Enhanced readable values function with better formatting
def get_readable_values(input_data):
    # Mapping dictionaries for encoded values (unchanged)
    gender_map = {0: "Male", 1: "Female"}
    yes_no_map = {0: "No", 1: "Yes"}
    veggie_map = {0: "Never", 1: "Sometimes", 2: "Always"}
    meals_map = {0: "1-2", 1: "3", 2: ">3"}
    frequency_map = {0: "Never", 1: "Sometimes", 2: "Frequently", 3: "Always"}
    water_map = {0: "Less than 1L", 1: "1-2L", 2: "More than 2L"}
    activity_map = {0: "Never", 1: "Once or twice a week", 2: "Two or three times a week", 3: "More than three times a week"}
    screen_map = {0: "None", 1: "Less than 1h", 2: "1-3h", 3: "More than 3h"}
    transport_map = {0: "Automobile", 1: "Public Transportation", 2: "Motorbike", 3: "Bike", 4: "Walking"}
    
    # Extract values safely using get() to handle potential index errors
    try:
        gender = gender_map.get(input_data[0], "Unknown")
        age = input_data[1] if 1 < len(input_data) else "Unknown"
        height = input_data[2] if 2 < len(input_data) else "Unknown"
        weight = input_data[3] if 3 < len(input_data) else "Unknown"
        family_history = yes_no_map.get(input_data[4], "Unknown") if 4 < len(input_data) else "Unknown"
        high_caloric_food = yes_no_map.get(input_data[5], "Unknown") if 5 < len(input_data) else "Unknown"
        veggie_freq = veggie_map.get(input_data[6], "Unknown") if 6 < len(input_data) else "Unknown"
        main_meals = meals_map.get(input_data[7], "Unknown") if 7 < len(input_data) else "Unknown"
        snacking = frequency_map.get(input_data[8], "Unknown") if 8 < len(input_data) else "Unknown"
        smoking = yes_no_map.get(input_data[9], "Unknown") if 9 < len(input_data) else "Unknown"
        water_intake = water_map.get(input_data[10], "Unknown") if 10 < len(input_data) else "Unknown"
        calorie_monitoring = yes_no_map.get(input_data[11], "Unknown") if 11 < len(input_data) else "Unknown"
        physical_activity = activity_map.get(input_data[12], "Unknown") if 12 < len(input_data) else "Unknown"
        screen_time = screen_map.get(input_data[13], "Unknown") if 13 < len(input_data) else "Unknown"
        alcohol = frequency_map.get(input_data[14], "Unknown") if 14 < len(input_data) else "Unknown"
        transportation = transport_map.get(input_data[15], "Unknown") if 15 < len(input_data) else "Unknown"
    except (IndexError, TypeError):
        # Handle potential errors with input_data
        return {feature: "Data Error" for feature in get_feature_names()}
    
    # BMI calculation (new)
    bmi = None
    if height not in ("Unknown", "Data Error") and weight not in ("Unknown", "Data Error"):
        try:
            # Convert height from cm to m for BMI calculation
            height_m = float(height) / 100
            bmi = float(weight) / (height_m * height_m)
            bmi = round(bmi, 1)
        except:
            bmi = None
    
    result = {
        "Gender": gender,
        "Age": age,
        "Height": f"{height} cm" if height not in ("Unknown", "Data Error") else height,
        "Weight": f"{weight} kg" if weight not in ("Unknown", "Data Error") else weight,
        "Family_History": family_history,
        "High_Caloric_Food": high_caloric_food,
        "Vegetable_Consumption": veggie_freq,
        "Main_Meals": main_meals,
        "Snacking": snacking,
        "Smoking": smoking,
        "Water_Intake": water_intake,
        "Calorie_Monitoring": calorie_monitoring,
        "Physical_Activity": physical_activity,
        "Screen_Time": screen_time,
        "Alcohol": alcohol,
        "Transportation": transportation
    }
    
    # Add BMI if available
    if bmi:
        result["BMI"] = bmi
    
    return result

# Generate enhanced text explanation based on SHAP values
def generate_text_explanation(shap_values, feature_names, readable_values):
    # Handle different types of shap_values
    if isinstance(shap_values, list):
        # For multi-class models, use positive class
        shap_values = shap_values[0]
    
    if len(shap_values.shape) > 1:
        shap_values = shap_values.flatten()
    
    # Ensure we only use valid indices
    valid_indices = [i for i in range(min(len(shap_values), len(feature_names)))]
    
    # Get indices of top features by absolute value (only from valid indices)
    top_indices = sorted(valid_indices, key=lambda i: abs(shap_values[i]), reverse=True)[:5]
    
    # Determine prediction direction
    risk_level = "HIGH" if sum(shap_values) > 0 else "LOW"
    risk_color = "#ef4444" if risk_level == "HIGH" else "#10b981"
    
    # Create explanation html with CSS classes for styling
    explanation = f"""
    <div class="explanation-card">
        <!-- Main prediction -->
        <div class="risk-header">
            <h2 style="margin: 0; color: {risk_color}; font-size: 24px;">
                Your obesity risk is <span style="font-weight: bold;">{risk_level}</span>
            </h2>
        </div>
        
        <h3 style="margin-top: 0; font-size: 18px; color: #333; text-align: center; margin-bottom: 20px;">
            Here's why our system made this prediction:
        </h3>
    """
    
    # Add BMI information with clear explanation if available
    if "BMI" in readable_values:
        bmi = readable_values["BMI"]
        bmi_category = "Unknown"
        bmi_color = "#666"
        bmi_explanation = ""
        
        if bmi < 18.5:
            bmi_category = "Underweight"
            bmi_color = "#3b82f6"
            bmi_explanation = "Being underweight typically lowers obesity risk, but can have other health concerns."
        elif 18.5 <= bmi < 25:
            bmi_category = "Normal weight"
            bmi_color = "#10b981"
            bmi_explanation = "Having a normal BMI is associated with lower obesity risk."
        elif 25 <= bmi < 30:
            bmi_category = "Overweight"
            bmi_color = "#f59e0b"
            bmi_explanation = "Being overweight increases your obesity risk. This is a key factor in your prediction."
        elif bmi >= 30:
            bmi_category = "Obese"
            bmi_color = "#ef4444"
            bmi_explanation = "Your BMI is in the obese range, which strongly indicates higher obesity risk."
        
        explanation += f"""
        <div class="bmi-box" style="border-left-color: {bmi_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 18px; font-weight: bold;">Your BMI is {bmi}</span>
                <span style="font-weight: bold; font-size: 18px; color: {bmi_color};">{bmi_category}</span>
            </div>
            <p style="margin: 0; font-size: 16px;">{bmi_explanation}</p>
            <p style="margin-top: 10px; font-size: 14px; color: #666;">BMI = weight(kg) ÷ height(m)²</p>
        </div>
        """
    
    # Add an introduction to the factors
    explanation += """
    <p style="font-size: 16px; margin-bottom: 20px;">
        These are the most important factors in your prediction:
    </p>
    """
    
    # Process each top factor
    for i, idx in enumerate(top_indices):
        feature = feature_names[idx]
        display_feature = feature.replace('_', ' ')
        value = readable_values.get(feature, "Unknown")
        impact = float(shap_values[idx])
        
        # Determine if this factor increases or decreases risk
        if impact > 0:
            direction = "INCREASES"
            icon = "⬆️"
            factor_class = "positive"  # Positive impact on risk (bad)
        else:   
            direction = "DECREASES"
            icon = "⬇️"
            factor_class = "negative"  # Negative impact on risk (good)
        
        # Create custom explanations for common factors
        factor_explanation = ""
        if feature == "Weight":
            if impact > 0:
                factor_explanation = f"Higher weight ({value}) is directly linked to increased obesity risk."
            else:
                factor_explanation = f"Your weight ({value}) helps lower your obesity risk."
                
        elif feature == "Age":
            if impact > 0:
                factor_explanation = f"At age {value}, your metabolism may be slowing, increasing obesity risk."
            else:
                factor_explanation = f"Your age ({value}) is associated with lower obesity risk in our model."
                
        elif feature == "Physical_Activity":
            if value in ["Never", "Once or twice a week"]:
                factor_explanation = "Low physical activity significantly increases obesity risk."
            else:
                factor_explanation = "Regular exercise helps prevent obesity by burning calories and building muscle."
                
        elif feature == "High_Caloric_Food":
            if value == "Yes":
                factor_explanation = "Regular consumption of high-calorie foods directly increases obesity risk."
            else:
                factor_explanation = "Avoiding high-calorie foods helps maintain a healthy weight."
                
        elif feature == "Vegetable_Consumption":
            if value == "Never":
                factor_explanation = "Not eating vegetables is linked to higher obesity risk."
            else:
                factor_explanation = "Eating vegetables helps maintain a healthy weight and provides essential nutrients."
                
        elif feature == "Family_History":
            if value == "Yes":
                factor_explanation = "Genetic factors from family history can predispose you to obesity."
            else:
                factor_explanation = "No family history of obesity reduces your genetic risk factors."
                
        elif feature == "Snacking":
            if value in ["Frequently", "Always"]:
                factor_explanation = "Frequent snacking can lead to consuming excess calories."
            else:
                factor_explanation = "Limited snacking helps control your total calorie intake."
                
        elif feature == "Water_Intake":
            if value == "More than 2L":
                factor_explanation = "Good hydration helps control appetite and supports metabolism."
            else:
                factor_explanation = "Increased water intake could help reduce your obesity risk."
                
        elif feature == "Screen_Time":
            if value in ["1-3h", "More than 3h"]:
                factor_explanation = "High screen time is associated with sedentary behavior and increased obesity risk."
            else:
                factor_explanation = "Limited screen time likely means you're more physically active."
        
        # Default explanation if nothing specific is defined
        if not factor_explanation:
            if impact > 0:
                factor_explanation = f"Your {display_feature.lower()} ({value}) contributes to higher obesity risk."
            else:
                factor_explanation = f"Your {display_feature.lower()} ({value}) helps reduce your obesity risk."
        
        explanation += f"""
        <div class="factor-box {factor_class}">
            <div class="factor-header">
                <span class="factor-icon">{icon}</span>
                <span class="factor-title {factor_class}">{display_feature}: {value}</span>
            </div>
            <p style="margin: 0; font-size: 16px;">{factor_explanation}</p>
            <p style="margin-top: 8px; font-size: 15px; font-weight: bold; color: {'#ef4444' if factor_class == 'positive' else '#10b981'}">
                This {direction.lower()} your obesity risk
            </p>
        </div>
        """
    
    # Add personalized recommendations based on the major factors
    explanation += """
    <div class="recommendations">
        <h3>What You Can Do</h3>
        <ul>
    """
    
    # Create custom recommendations based on the top factors
    recommendations = []
    for idx in top_indices:
        feature = feature_names[idx]
        impact = float(shap_values[idx])
        
        # Only create recommendations for factors that increase risk
        if impact <= 0:
            continue
            
        if feature == "Weight":
            recommendations.append("Work with a healthcare provider to develop a healthy weight loss plan")
        elif feature == "Physical_Activity":
            recommendations.append("Aim for at least 150 minutes of moderate exercise each week")
        elif feature == "High_Caloric_Food":
            recommendations.append("Reduce consumption of fast food, fried foods, and sugary snacks")
        elif feature == "Vegetable_Consumption":
            recommendations.append("Add more vegetables to your meals - aim for half your plate to be vegetables")
        elif feature == "Snacking":
            recommendations.append("Replace unhealthy snacks with fruits, vegetables, or nuts")
        elif feature == "Water_Intake":
            recommendations.append("Drink more water throughout the day (aim for 2L or more)")
        elif feature == "Screen_Time":
            recommendations.append("Take breaks from screens to walk or stretch every hour")
        elif feature == "Main_Meals":
            recommendations.append("Focus on regular, balanced meals instead of skipping or overeating")
        elif feature == "Alcohol":
            recommendations.append("Reduce alcohol consumption, which contains empty calories")
        elif feature == "Transportation":
            recommendations.append("Look for opportunities to walk or bike instead of driving when possible")
    
    # Add default recommendation if none were generated
    if not recommendations:
        recommendations.append("Focus on maintaining a balanced diet and regular physical activity")
        recommendations.append("Consult with a healthcare provider for personalized health advice")
    
    # Add the recommendations to the explanation
    for recommendation in recommendations:
        explanation += f"""
        <li>{recommendation}</li>
        """
    
    explanation += """
        </ul>
    </div>
    """
    
    
    explanation += "</div>"
    
    return explanation


def create_health_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Health Score", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': get_color_from_score(score)},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#fee2e2'},
                {'range': [40, 70], 'color': '#fef9c3'},
                {'range': [70, 100], 'color': '#dcfce7'}
            ],
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Get color based on health score
def get_color_from_score(score):
    if score < 40:
        return "#ef4444"  # red
    elif score < 70:
        return "#f59e0b"  # amber
    else:
        return "#10b981"  # green

# Calculate BMI color based on value
def get_bmi_color(bmi):
    if bmi < 18.5:
        return "#f59e0b"  # Underweight - amber
    elif 18.5 <= bmi < 25:
        return "#10b981"  # Normal - green
    elif 25 <= bmi < 30:
        return "#f59e0b"  # Overweight - amber
    else:
        return "#ef4444"  # Obese - red

# Calculate health score from user inputs
def calculate_health_score(bmi, activity_encoded, vegetable_encoded, high_caloric_food_encoded, 
                          calorie_monitoring_encoded, smoking_encoded, alcohol_encoded, 
                          screen_encoded, family_history_encoded, water_encoded):
    # Base score
    score = 70
    
    # BMI impact
    if bmi < 18.5:  # Underweight
        score -= 5
    elif 18.5 <= bmi < 25:  # Normal
        score += 10
    elif 25 <= bmi < 30:  # Overweight
        score -= 5
    else:  # Obese
        score -= 15
    
    # Activity impact
    if activity_encoded == 0:
        score -= 10
    elif activity_encoded == 1:
        score -= 5
    elif activity_encoded == 2:
        score += 5
    elif activity_encoded >= 3:
        score += 10
    
    # Diet impact
    score += (vegetable_encoded * 3)
    score -= (8 if high_caloric_food_encoded else 0)
    score += (5 if calorie_monitoring_encoded else 0)
    score += (water_encoded * 2)
    
    # Risk factors
    score -= (15 if smoking_encoded else 0)
    score -= (alcohol_encoded * 3)
    score -= (screen_encoded * 2)
    score -= (5 if family_history_encoded else 0)
    
    # Ensure score is between 0 and 100
    return max(0, min(100, score))

# Generate health achievements based on metrics
def get_achievements(bmi, activity_encoded, vegetable_encoded, water_encoded, calorie_monitoring_encoded):
    achievements = []
    
    if 18.5 <= bmi < 25:
        achievements.append(("Healthy BMI", "Maintaining a BMI in the healthy range"))
        
    if activity_encoded >= 2:
        achievements.append(("Active Lifestyle", "Regular physical activity"))
        
    if vegetable_encoded >= 2:
        achievements.append(("Nutrition Expert", "High vegetable consumption"))
        
    if water_encoded >= 2:
        achievements.append(("Well Hydrated", "Drinking enough water daily"))
        
    if calorie_monitoring_encoded:
        achievements.append(("Mindful Eater", "Monitoring calorie intake"))
    
    return achievements

# Create mini charts for tracking progress
def create_progress_chart(current, target, title, color):
    progress = min(100, (current / target) * 100)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = progress,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 14}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 0, 'visible': False},
            'bar': {'color': color},
            'bgcolor': "#e2e8f0",
            'borderwidth': 0,
        },
        number = {'suffix': "%"}
    ))
    
    fig.update_layout(
        height=120,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Main application header
st.markdown('<h1 class="main-header">Obesity Risk Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subheader">Assess your risk factors and get personalized health insights</p>', unsafe_allow_html=True)

# Load model
model = load_model()

# Create tabs for different app sections
tabs = st.tabs(["Risk Assessment", "Health Profile", "Model Insights"])

with tabs[0]:
    st.markdown('<h2 class="subheader">Personal Risk Assessment</h2>', unsafe_allow_html=True)
    
    # Create two main columns for the entire layout
    left_col, right_col = st.columns([1, 1])
    
    # Left column for input form
    with left_col:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            st.markdown('<p class="section-title">Demographics</p>', unsafe_allow_html=True)
            gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
            age = st.number_input("Age", min_value=10, max_value=100, value=30)
            height = st.number_input("Height (cm)", min_value=120, max_value=220, value=170)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)
            
            st.markdown('<p class="section-title">Health History</p>', unsafe_allow_html=True)
            family_history = st.selectbox("Family History of Obesity", options=["No", "Yes"], index=0)
            smoking = st.selectbox("Do you smoke?", options=["No", "Yes"], index=0)
            alcohol = st.selectbox("Alcohol Consumption", options=["Never", "Sometimes", "Frequently", "Always"], index=1)
            
            st.markdown('<p class="section-title">Dietary Habits</p>', unsafe_allow_html=True)
            high_caloric_food = st.selectbox("High Caloric Food Consumption", options=["No", "Yes"], index=0)
            vegetable_consumption = st.selectbox("Vegetable Consumption", options=["Never", "Sometimes", "Always"], index=1)
            main_meals = st.selectbox("Number of Main Meals", options=["1-2", "3", ">3"], index=1)
            snacking = st.selectbox("Frequency of Snacking", options=["Never", "Sometimes", "Frequently", "Always"], index=1)
            water_intake = st.selectbox("Daily Water Intake", options=["Less than 1L", "1-2L", "More than 2L"], index=1)
            calorie_monitoring = st.selectbox("Do you monitor calories?", options=["No", "Yes"], index=0)
            
            st.markdown('<p class="section-title">Lifestyle</p>', unsafe_allow_html=True)
            physical_activity = st.selectbox("Physical Activity Frequency", 
                                      options=["Never", "Once or twice a week", "Two or three times a week", "More than three times a week"], 
                                      index=1)
            screen_time = st.selectbox("Daily Screen Time", options=["None", "Less than 1h", "1-3h", "More than 3h"], index=2)
            transportation = st.selectbox("Primary Mode of Transportation", 
                                  options=["Automobile", "Public Transportation", "Motorbike", "Bike", "Walking"], 
                                  index=0)
                        
            # Create a button to predict
            predict_button = st.button("Predict Obesity Risk", type="primary")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Right column for results and explanations
    with right_col:
        # Process prediction when button is clicked
        if predict_button and model:
            # Transform inputs into model format
            gender_encoded = 0 if gender == "Male" else 1
            family_history_encoded = 0 if family_history == "No" else 1
            high_caloric_food_encoded = 0 if high_caloric_food == "No" else 1
            vegetable_map = {"Never": 0, "Sometimes": 1, "Always": 2}
            vegetable_encoded = vegetable_map[vegetable_consumption]
            meals_map = {"1-2": 0, "3": 1, ">3": 2}
            meals_encoded = meals_map[main_meals]
            frequency_map = {"Never": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
            snacking_encoded = frequency_map[snacking]
            smoking_encoded = 0 if smoking == "No" else 1
            water_map = {"Less than 1L": 0, "1-2L": 1, "More than 2L": 2}
            water_encoded = water_map[water_intake]
            calorie_monitoring_encoded = 0 if calorie_monitoring == "No" else 1
            activity_map = {"Never": 0, "Once or twice a week": 1, "Two or three times a week": 2, "More than three times a week": 3}
            activity_encoded = activity_map[physical_activity]
            screen_map = {"None": 0, "Less than 1h": 1, "1-3h": 2, "More than 3h": 3}
            screen_encoded = screen_map[screen_time]
            alcohol_encoded = frequency_map[alcohol]
            transport_map = {"Automobile": 0, "Public Transportation": 1, "Motorbike": 2, "Bike": 3, "Walking": 4}
            transport_encoded = transport_map[transportation]
            
            # Create input array for model
            input_data = [
                gender_encoded, age, height, weight, family_history_encoded, 
                high_caloric_food_encoded, vegetable_encoded, meals_encoded, 
                snacking_encoded, smoking_encoded, water_encoded, calorie_monitoring_encoded, 
                activity_encoded, screen_encoded, alcohol_encoded, transport_encoded
            ]
            
            # Predict obesity category
            prediction, category = predict_obesity_level(model, input_data)
            
            # Calculate BMI
            bmi = calculate_bmi(weight, height)
            bmi_category, bmi_color = get_bmi_category(bmi)
            
            # Display results
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Create two columns for prediction and BMI display
            results_col1, results_col2 = st.columns(2)
            
            with results_col1:
                st.markdown(f'<h3 class="section-title">Predicted Obesity Category</h3>', unsafe_allow_html=True)
                category_color = get_category_color(category)
                st.markdown(f'<div class="prediction-result" style="background-color: {category_color}20; border: 1px solid {category_color};">'
                          f'<h2 style="color: {category_color}">Prediction Result</h2>'
                          f'<h1 style="color: {category_color}">{category}</h1>'
                          f'</div>', unsafe_allow_html=True)
                
                # Display category chart
                category_chart = create_category_chart(category)
                st.plotly_chart(category_chart, use_container_width=True)
            
            with results_col2:
                st.markdown(f'<h3 class="section-title">Body Mass Index (BMI)</h3>', unsafe_allow_html=True)
                st.markdown(f'<div class="prediction-result" style="background-color: {bmi_color}20; border: 1px solid {bmi_color};">'
                          f'<h2 style="color: {bmi_color}">BMI Category</h2>'
                          f'<h1 style="color: {bmi_color}">{bmi_category} ({bmi})</h1>'
                          f'</div>', unsafe_allow_html=True)
                
                # Display BMI gauge
                bmi_gauge = create_bmi_gauge(bmi)
                st.plotly_chart(bmi_gauge, use_container_width=True)
            
            # Add a divider for risk factors analysis
            st.markdown('<div class="section-divider"><div class="section-divider-line"></div>'
                      '<div class="section-divider-text">Risk Factors Analysis</div>'
                      '<div class="section-divider-line"></div></div>', unsafe_allow_html=True)
            
            # SHAP Analysis
            if model:
                try:
                    st.markdown('<h2 class="subheader">Explanation of Prediction</h2>', unsafe_allow_html=True)

                    # Create SHAP explainer
                    with st.spinner("Generating SHAP explanations..."):
                        # Verify model type is compatible with TreeExplainer
                        if not hasattr(model, 'predict'):
                            st.error("The model doesn't have a predict method and may not be compatible with TreeExplainer")
                            raise ValueError("Incompatible model type")
                            
                        explainer = shap.TreeExplainer(model)
                        
                        # Convert input data to numpy array
                        input_array = np.array(input_data).reshape(1, -1)
                        
                        # Get feature names
                        try:
                            feature_names = get_feature_names()
                        except Exception as e:
                            st.warning(f"Error getting feature names: {str(e)}")
                            # Fallback to generic feature names
                            feature_names = [f"Feature {i}" for i in range(input_array.shape[1])]
                        
                        # Calculate SHAP values with input
                        try:
                            shap_values = explainer.shap_values(input_array)
                        except Exception as e:
                            st.error(f"Error calculating SHAP values: {str(e)}")
                            raise
                        
                        # Handle different SHAP value formats and save class_index for later use
                        if isinstance(shap_values, list):
                            # For multi-class models
                            try:
                                prediction_value = float(prediction)
                            except (ValueError, TypeError):
                                st.warning("Invalid prediction value, defaulting to class 0")
                                prediction_value = 0
                            
                            # Ensure class_index is valid
                            class_index = min(int(prediction_value), len(shap_values)-1)
                            class_index = max(0, class_index)  # Ensure it's not negative
                            
                            shap_values_for_instance = shap_values[class_index][0]
                        else:
                            # For binary classification or regression
                            shap_values_for_instance = shap_values[0]
                            class_index = 0  # Set default for non-multi-class case
                        
                        # Ensure we have a flat array
                        shap_values_for_instance = np.array(shap_values_for_instance).flatten()
                        
                        # Handle the dimension mismatch
                        if len(shap_values_for_instance) != len(feature_names):
                            # If there are more SHAP values than features, it might be one-hot encoded
                            if len(shap_values_for_instance) > len(feature_names):
                                # Approach 1: Sum up SHAP values for each feature (assuming one-hot encoding)
                                # This is just a heuristic approach
                                values_per_feature = len(shap_values_for_instance) // len(feature_names)
                                if values_per_feature * len(feature_names) == len(shap_values_for_instance):
                                    aggregated_values = []
                                    for i in range(len(feature_names)):
                                        start_idx = i * values_per_feature
                                        end_idx = start_idx + values_per_feature
                                        aggregated_values.append(np.sum(shap_values_for_instance[start_idx:end_idx]))
                                    shap_values_for_instance = np.array(aggregated_values)
                                else:
                                    # Can't determine a clean division, use top features by magnitude
                                    st.warning("Cannot determine encoding pattern. Using top features by importance.")
                                    # Sort by absolute value and take top N
                                    top_indices = np.argsort(np.abs(shap_values_for_instance))[::-1][:len(feature_names)]
                                    shap_values_for_instance = shap_values_for_instance[top_indices]
                            else:
                                # More feature names than SHAP values
                                # Take only the first N feature names where N is the length of SHAP values
                                feature_names = feature_names[:len(shap_values_for_instance)]
                        
                        # Get readable values for explanation
                        try:
                            readable_values = get_readable_values(input_data)
                        except Exception as e:
                            st.warning(f"Error getting readable values: {str(e)}")
                            readable_values = {f: str(input_array[0, i]) for i, f in enumerate(feature_names)}
                        
                        # Display text explanation
                        try:
                            explanation = generate_text_explanation(shap_values_for_instance, feature_names, readable_values)
                            # Use streamlit components.html to render the HTML
                            with st.container(height=550):  # Increased height to accommodate the enhanced explanation
                                st.components.v1.html(f"""
                                <style>
                                /* Base container styling */
                                .explanation-card {{
                                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                                    border: 1px solid #e1e4e8;
                                    border-radius: 16px;
                                    padding: 24px;
                                    margin: 0 auto;
                                    background-color: white;
                                    box-shadow: 0 6px 18px rgba(0,0,0,0.1);
                                    overflow-y: auto;
                                    max-height: 520px;
                                    transition: all 0.3s ease;
                                }}
                                
                                .explanation-card:hover {{
                                    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
                                }}
                                
                                /* Risk level header */
                                .risk-header {{
                                    text-align: center;
                                    margin-bottom: 28px;
                                    padding: 18px;
                                    background-color: #f8f9fa;
                                    border-radius: 12px;
                                    border-left: 5px solid #3498db;
                                    transition: all 0.3s ease;
                                }}
                                
                                .risk-header:hover {{
                                    background-color: #e9f7fe;
                                }}
                                
                                /* Factor box styles */
                                .factor-box {{
                                    margin-bottom: 22px;
                                    padding: 18px;
                                    border-radius: 12px;
                                    border-left-width: 5px;
                                    border-left-style: solid;
                                    transition: all 0.3s ease;
                                    position: relative;
                                    overflow: hidden;
                                }}
                                
                                .factor-box::after {{
                                    content: '';
                                    position: absolute;
                                    bottom: 0;
                                    right: 0;
                                    width: 40px;
                                    height: 40px;
                                    border-radius: 50% 0 0 0;
                                    opacity: 0.1;
                                    z-index: 0;
                                }}
                                
                                .factor-box.positive {{
                                    background-color: #fff5f5;
                                    border-left-color: #ef4444;
                                }}
                                
                                .factor-box.positive::after {{
                                    background-color: #ef4444;
                                }}
                                
                                .factor-box.negative {{
                                    background-color: #f0fff4;
                                    border-left-color: #10b981;
                                }}
                                
                                .factor-box.negative::after {{
                                    background-color: #10b981;
                                }}
                                
                                .factor-box:hover {{
                                    transform: translateY(-2px);
                                    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                                }}
                                
                                /* Factor header with icon */
                                .factor-header {{
                                    display: flex;
                                    align-items: center;
                                    margin-bottom: 12px;
                                    position: relative;
                                    z-index: 1;
                                }}
                                
                                .factor-icon {{
                                    font-size: 22px;
                                    margin-right: 10px;
                                    width: 32px;
                                    height: 32px;
                                    border-radius: 50%;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    color: white;
                                }}
                                
                                .factor-icon.positive {{
                                    background-color: #ef4444;
                                }}
                                
                                .factor-icon.negative {{
                                    background-color: #10b981;
                                }}
                                
                                .factor-title {{
                                    font-weight: 600;
                                    font-size: 18px;
                                }}
                                
                                .factor-title.positive {{
                                    color: #ef4444;
                                }}
                                
                                .factor-title.negative {{
                                    color: #10b981;
                                }}
                                
                                .factor-content {{
                                    position: relative;
                                    z-index: 1;
                                    font-size: 16px;
                                    line-height: 1.6;
                                }}
                                
                                /* BMI box styling */
                                .bmi-box {{
                                    background-color: #f0f5ff;
                                    border-radius: 12px;
                                    padding: 18px;
                                    margin-bottom: 28px;
                                    border-left: 5px solid #3b82f6;
                                    transition: all 0.3s ease;
                                }}
                                
                                .bmi-box:hover {{
                                    background-color: #e6effd;
                                    transform: translateY(-2px);
                                    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                                }}
                                
                                /* Recommendations section */
                                .recommendations {{
                                    margin-top: 28px;
                                    background-color: #f8f9fa;
                                    border-radius: 12px;
                                    padding: 18px;
                                    border-left: 5px solid #3b82f6;
                                    transition: all 0.3s ease;
                                }}
                                
                                .recommendations:hover {{
                                    background-color: #e6effd;
                                    transform: translateY(-2px);
                                    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                                }}
                                
                                .recommendations h3 {{
                                    margin-top: 0;
                                    color: #3b82f6;
                                    font-size: 20px;
                                    display: flex;
                                    align-items: center;
                                }}
                                
                                .recommendations h3::before {{
                                    content: '💡';
                                    margin-right: 8px;
                                }}
                                
                                .recommendations ul {{
                                    margin-top: 12px;
                                    padding-left: 28px;
                                }}
                                
                                .recommendations li {{
                                    margin-bottom: 10px;
                                    font-size: 16px;
                                    position: relative;
                                }}
                                
                                .recommendations li::before {{
                                    content: '✓';
                                    position: absolute;
                                    left: -20px;
                                    color: #3b82f6;
                                    font-weight: bold;
                                }}
                                
                                /* Disclaimer */
                                .disclaimer {{
                                    margin-top: 24px;
                                    font-size: 14px;
                                    color: #666;
                                    text-align: center;
                                    padding: 10px;
                                    background-color: #f8f9fa;
                                    border-radius: 8px;
                                }}
                                
                                /* Responsive adjustments */
                                @media (max-width: 768px) {{
                                    .explanation-card {{
                                        padding: 16px;
                                    }}
                                    
                                    .factor-title {{
                                        font-size: 16px;
                                    }}
                                    
                                    .factor-content {{
                                        font-size: 14px;
                                    }}
                                }}
                                </style>
                                {explanation}
                                """, height=520)
                        except Exception as e:
                            st.error(f"Error generating text explanation: {str(e)}")
                            # Show simple explanation as fallback
                            top_features = sorted(zip(feature_names, shap_values_for_instance), 
                                                key=lambda x: abs(x[1]), reverse=True)[:5]
                            st.markdown("### Top influencing features:")
                            for feature, value in top_features:
                                direction = "increases" if value > 0 else "decreases"
                                st.markdown(f"- **{feature}**: {direction} risk by {abs(value):.2f}")
                                
                        # SHAP visualization tabs
                        shap_tabs = st.tabs(["Force Plot", "Waterfall Plot", "Feature Importance", "Decision Plot"])
                        
                        with shap_tabs[0]:
                            st.markdown("#### Force Plot")
                            st.markdown("Shows how each feature pushes the prediction from the base value.")
                            
                            try:
                                # Fix for force plot - instead of using the create_shap_force_plot function
                                # Create force plot directly
                                import matplotlib.pyplot as plt
                                
                                plt.figure(figsize=(10, 3))
                                # Use shap's force_plot and capture as HTML, or use a different visualization
                                # Option 1: Use matplotlib to create a similar visualization
                                feature_importance = np.abs(shap_values_for_instance)
                                sorted_idx = np.argsort(feature_importance)
                                plt.barh(range(len(sorted_idx)), shap_values_for_instance[sorted_idx], 
                                        color=['red' if x > 0 else 'blue' for x in shap_values_for_instance[sorted_idx]])
                                plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
                                plt.title('SHAP Force Plot (Alternative)')
                                plt.tight_layout()
                                st.pyplot(plt)
                                
                                # Option 2 (if possible): Use shap's plotting directly
                                # This is a cleaner approach if it works with your SHAP version
                                try:
                                    # For newer SHAP versions
                                    plt.figure()
                                    shap.plots.waterfall(shap_values_for_instance, max_display=10, show=False)
                                    st.pyplot(plt)
                                except:
                                    pass
                            except Exception as e:
                                st.error(f"Error creating force plot: {str(e)}")
                            
                        with shap_tabs[1]:
                            st.markdown("#### Waterfall Plot")
                            st.markdown("Visualizes how each feature contributes to push the model output from the base value to the final prediction.")
                            
                            try:
                                # Direct waterfall implementation
                                plt.figure(figsize=(10, 6))
                                # Sort indices by magnitude
                                indices = np.argsort(np.abs(shap_values_for_instance))[::-1][:10]  # Top 10 features
                                
                                # Plot waterfall
                                cumulative = np.zeros(len(indices) + 1)
                                cumulative[1:] = np.cumsum(shap_values_for_instance[indices])
                                base_value = explainer.expected_value
                                
                                # Handle different formats of expected_value
                                if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
                                    try:
                                        base_value = base_value[class_index]
                                    except:
                                        base_value = base_value[0]  # Default to first class if class_index is invalid
                                
                                # Add base value
                                plt.barh(0, base_value, color='gray')
                                plt.text(base_value, 0, f'Base: {base_value:.2f}', ha='left', va='center')
                                
                                # Add each feature's contribution
                                for i, idx in enumerate(indices):
                                    plt.barh(i+1, shap_values_for_instance[idx], 
                                            left=cumulative[i], 
                                            color='red' if shap_values_for_instance[idx] > 0 else 'blue')
                                    plt.text(cumulative[i+1], i+1, 
                                            f'{feature_names[idx]}: {shap_values_for_instance[idx]:.2f}', 
                                            ha='left' if shap_values_for_instance[idx] > 0 else 'right', 
                                            va='center')
                                
                                # Final prediction
                                plt.barh(len(indices)+1, 0, left=cumulative[-1] + base_value, color='gray')
                                plt.text(cumulative[-1] + base_value, len(indices)+1, 
                                        f'Final: {cumulative[-1] + base_value:.2f}', 
                                        ha='left', va='center')
                                
                                plt.yticks(range(len(indices) + 2), 
                                        ['Base Value'] + [feature_names[i] for i in indices] + ['Final Prediction'])
                                plt.title('Waterfall Plot')
                                plt.grid(axis='x', linestyle='--', alpha=0.7)
                                plt.tight_layout()
                                st.pyplot(plt)
                            except Exception as e:
                                st.error(f"Error creating waterfall plot: {str(e)}")
                            
                        with shap_tabs[2]:
                            st.markdown("#### Feature Importance")
                            st.markdown("Shows which features are most important for this prediction.")
                            
                            try:
                                # Create a simple bar chart directly
                                plt.figure(figsize=(10, 6))
                                importances = np.abs(shap_values_for_instance)
                                indices = np.argsort(importances)[::-1]
                                plt.barh(range(len(indices)), importances[indices], color='skyblue')
                                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                                plt.xlabel('Absolute SHAP Value (Feature Importance)')
                                plt.title('Feature Importance')
                                plt.tight_layout()
                                st.pyplot(plt)
                            except Exception as e:
                                st.error(f"Error creating importance plot: {str(e)}")
                            
                        with shap_tabs[3]:
                            st.markdown("#### Decision Plot")
                            st.markdown("Shows the path from the base value to the final prediction.")
                            
                            try:
                                # Create an alternative decision plot
                                plt.figure(figsize=(10, 6))
                                # Sort values by importance
                                sorted_idx = np.argsort(np.abs(shap_values_for_instance))[::-1][:10]  # Top 10
                                
                                # Get the expected value
                                base_value = explainer.expected_value
                                
                                # Handle different formats of expected_value
                                if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
                                    try:
                                        base_value = base_value[class_index]
                                    except:
                                        base_value = base_value[0]  # Default to first class if class_index is invalid
                                    
                                # Calculate cumulative values
                                sorted_values = shap_values_for_instance[sorted_idx]
                                cum_values = np.cumsum(sorted_values)
                                cum_values = np.insert(cum_values, 0, 0)  # Start from 0
                                
                                # Plot
                                for i in range(1, len(cum_values)):
                                    plt.plot([i-1, i], [base_value + cum_values[i-1], base_value + cum_values[i]], 
                                            marker='o', color='blue' if sorted_values[i-1] > 0 else 'red')
                                
                                plt.axhline(y=base_value, color='gray', linestyle='--', label='Base Value')
                                plt.text(0, base_value, f'Base: {base_value:.2f}', va='bottom', ha='left')
                                
                                # Final value
                                plt.text(len(sorted_idx), base_value + cum_values[-1], 
                                        f'Final: {base_value + cum_values[-1]:.2f}', va='bottom', ha='right')
                                
                                plt.xticks(range(len(sorted_idx)), 
                                        [feature_names[i] for i in sorted_idx], rotation=45, ha='right')
                                plt.title('Decision Plot: Path to Prediction')
                                plt.xlabel('Features (ordered by importance)')
                                plt.ylabel('Prediction Value')
                                plt.grid(True, linestyle='--', alpha=0.7)
                                plt.tight_layout()
                                st.pyplot(plt)
                            except Exception as e:
                                st.error(f"Error creating decision plot: {str(e)}")
                
                except Exception as e:
                    st.error(f"Error in SHAP analysis: {str(e)}")
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")
            
            st.markdown('</div>', unsafe_allow_html=True)
with tabs[1]:
    # Load custom CSS
    load_css()
    
    st.markdown('<h2 class="subheader">Your Health Profile</h2>', unsafe_allow_html=True)
    
    if 'prediction' not in locals():
        # Enhanced placeholder with animation
        st.markdown('<div class="placeholder-content">'
                  '<div class="placeholder-icon">📊</div>'
                  '<div class="placeholder-text">Complete the Risk Assessment first to view your Health Profile</div>'
                  '<div style="margin-top:10px; font-size:0.9rem; color:#94a3b8;">Fill out the form on the Assessment tab to generate your personalized health insights</div>'
                  '</div>', unsafe_allow_html=True)
    else:
        # Calculate health score
        health_score = calculate_health_score(bmi, activity_encoded, vegetable_encoded, 
                                             high_caloric_food_encoded, calorie_monitoring_encoded, 
                                             smoking_encoded, alcohol_encoded, screen_encoded, 
                                             family_history_encoded, water_encoded)
        
        # Get health achievements
        achievements = get_achievements(bmi, activity_encoded, vegetable_encoded, water_encoded, calorie_monitoring_encoded)
        
        # Health Score visualization
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Your Health Score</h3>', unsafe_allow_html=True)
        
        score_col1, score_col2 = st.columns([1, 2])
        
        with score_col1:
            score_gauge = create_health_gauge(health_score)
            st.plotly_chart(score_gauge, use_container_width=True, config={'displayModeBar': False})
        
        with score_col2:
            score_text = "Needs Improvement" if health_score < 40 else "Good" if health_score < 70 else "Excellent"
            score_color = get_color_from_score(health_score)
            
            st.markdown(f'<div style="margin-bottom:15px;">'
                      f'<div style="font-size:1.25rem; font-weight:600; margin-bottom:5px;">Your health is <span style="color:{score_color}">{score_text}</span></div>'
                      f'<div style="color:#64748b;">Based on your assessment responses</div>'
                      f'</div>', unsafe_allow_html=True)
            
            # Display achievements
            if achievements:
                st.markdown('<div style="margin-top:15px; margin-bottom:10px; font-weight:500;">Your achievements:</div>', unsafe_allow_html=True)
                for achievement, description in achievements[:3]:  # Limit to 3
                    st.markdown(f'<div class="badge badge-success" title="{description}">{achievement}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display health metrics with enhanced visuals
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Key Health Metrics</h3>', unsafe_allow_html=True)
        
        # Create metric containers
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        
        # BMI metric with tooltip
        bmi_color = get_bmi_color(bmi)
        bmi_status = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
        
        st.markdown(f'<div class="metric-item">'
                  f'<div class="metric-value" style="color: {bmi_color}">{bmi:.1f}</div>'
                  f'<div class="metric-label">BMI</div>'
                  f'<div class="tooltip metric-info">ⓘ'
                  f'<span class="tooltiptext">Body Mass Index: {bmi_status}<br>Healthy range: 18.5-24.9</span>'
                  f'</div>'
                  f'</div>', unsafe_allow_html=True)
        
        # Age metric
        st.markdown(f'<div class="metric-item">'
                  f'<div class="metric-value" style="color: #3b82f6">{age}</div>'
                  f'<div class="metric-label">Age</div>'
                  f'</div>', unsafe_allow_html=True)
        
        # Height metric
        st.markdown(f'<div class="metric-item">'
                  f'<div class="metric-value" style="color: #8b5cf6">{height} cm</div>'
                  f'<div class="metric-label">Height</div>'
                  f'</div>', unsafe_allow_html=True)
        
        # Weight metric
        st.markdown(f'<div class="metric-item">'
                  f'<div class="metric-value" style="color: #ec4899">{weight} kg</div>'
                  f'<div class="metric-label">Weight</div>'
                  f'<div class="tooltip metric-info">ⓘ'
                  f'<span class="tooltiptext">Ideal weight range: {int(18.5 * (height/100)**2)}-{int(24.9 * (height/100)**2)} kg</span>'
                  f'</div>'
                  f'</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Lifestyle summary with enhanced visuals
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Lifestyle Summary</h3>', unsafe_allow_html=True)
        
        # Create columns for lifestyle factors
        ls_col1, ls_col2, ls_col3 = st.columns(3)
        
        with ls_col1:
            st.markdown('<p class="section-title">Physical Activity</p>', unsafe_allow_html=True)
            activity_color = "#ef4444" if activity_encoded < 2 else "#10b981"
            activity_progress = min(100, (activity_encoded / 3) * 100) 
            
            st.markdown(f'<div style="padding: 16px; border-radius: 8px; background-color: {activity_color}20; border: 1px solid {activity_color};">'
                      f'<div style="font-weight: 600; font-size: 1.1rem; color: {activity_color}">{physical_activity}</div>'
                      f'<div style="font-size: 0.9rem; color: #64748b; margin-top: 4px;">Activity Level</div>'
                      f'<div style="height: 6px; background-color: #e2e8f0; border-radius: 3px; margin-top: 10px;">'
                      f'<div style="height: 6px; width: {activity_progress}%; background-color: {activity_color}; border-radius: 3px;"></div>'
                      f'</div>'
                      f'</div>', unsafe_allow_html=True)
        
        with ls_col2:
            st.markdown('<p class="section-title">Diet Quality</p>', unsafe_allow_html=True)
            diet_score = vegetable_encoded * 2 - (1 if high_caloric_food_encoded else 0) + (1 if calorie_monitoring_encoded else 0)
            diet_status = "Needs Improvement" if diet_score < 2 else "Good" if diet_score < 4 else "Excellent"
            diet_color = "#ef4444" if diet_score < 2 else "#f59e0b" if diet_score < 4 else "#10b981"
            diet_progress = min(100, (diet_score / 6) * 100)
            
            st.markdown(f'<div style="padding: 16px; border-radius: 8px; background-color: {diet_color}20; border: 1px solid {diet_color};">'
                      f'<div style="font-weight: 600; font-size: 1.1rem; color: {diet_color}">{diet_status}</div>'
                      f'<div style="font-size: 0.9rem; color: #64748b; margin-top: 4px;">Diet Quality</div>'
                      f'<div style="height: 6px; background-color: #e2e8f0; border-radius: 3px; margin-top: 10px;">'
                      f'<div style="height: 6px; width: {diet_progress}%; background-color: {diet_color}; border-radius: 3px;"></div>'
                      f'</div>'
                      f'</div>', unsafe_allow_html=True)
        
        with ls_col3:
            st.markdown('<p class="section-title">Risk Factors</p>', unsafe_allow_html=True)
            risk_count = family_history_encoded + smoking_encoded + (1 if alcohol_encoded > 1 else 0) + (1 if screen_encoded > 2 else 0)
            risk_status = "Low" if risk_count < 1 else "Moderate" if risk_count < 3 else "High"
            risk_color = "#10b981" if risk_count < 1 else "#f59e0b" if risk_count < 3 else "#ef4444"
            risk_progress = min(100, (risk_count / 4) * 100)
            
            st.markdown(f'<div style="padding: 16px; border-radius: 8px; background-color: {risk_color}20; border: 1px solid {risk_color};">'
                      f'<div style="font-weight: 600; font-size: 1.1rem; color: {risk_color}">{risk_status}</div>'
                      f'<div style="font-size: 0.9rem; color: #64748b; margin-top: 4px;">Risk Level</div>'
                      f'<div style="height: 6px; background-color: #e2e8f0; border-radius: 3px; margin-top: 10px;">'
                      f'<div style="height: 6px; width: {risk_progress}%; background-color: {risk_color}; border-radius: 3px;"></div>'
                      f'</div>'
                      f'</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced recommendations
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Personalized Recommendations</h3>', unsafe_allow_html=True)
        
        # Same recommendations logic as before but with enhanced display
        recommendations = []
        
        # Diet recommendations
        if vegetable_encoded < 2:
            recommendations.append("Increase your vegetable consumption to at least 5 servings per day")
        
        if high_caloric_food_encoded:
            recommendations.append("Reduce consumption of high-calorie processed foods")
        
        if water_encoded < 2:
            recommendations.append("Increase water intake to more than 2 liters per day")
        
        if not calorie_monitoring_encoded:
            recommendations.append("Consider monitoring your calorie intake to better understand your diet")
        
        # Activity recommendations
        if activity_encoded < 2:
            recommendations.append("Increase physical activity to at least 2-3 times per week")
        
        if screen_encoded > 2:
            recommendations.append("Reduce daily screen time to less than 2 hours")
        
        if transport_encoded < 3:
            recommendations.append("Consider active transportation like biking or walking when possible")
        
        # Health risk recommendations
        if smoking_encoded:
            recommendations.append("Quitting smoking will significantly improve your health")
        
        if alcohol_encoded > 1:
            recommendations.append("Reduce alcohol consumption")
        
        if family_history_encoded:
            recommendations.append("With family history of obesity, regular health checkups are recommended")
        
        # Add general recommendations if few specific ones
        if len(recommendations) < 3:
            recommendations.append("Maintain a balanced diet rich in fruits, vegetables, and lean proteins")
            recommendations.append("Aim for at least 150 minutes of moderate activity per week")
            recommendations.append("Ensure adequate sleep of 7-9 hours each night")
        
        # Display recommendations with enhanced styling
        for i, rec in enumerate(recommendations[:6]):  # Limit to 6 recommendations
            st.markdown(f'<div class="recommendation-item">'
                      f'<div class="recommendation-number">{i+1}</div>'
                      f'<div class="recommendation-text">{rec}</div>'
                      f'</div>', unsafe_allow_html=True)
        
        # Action plan section
        st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<h4 style="font-size: 1.1rem; font-weight: 600; color: #334155; margin-bottom: 10px;">Next Steps Action Plan</h4>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div style="padding: 12px; border-radius: 8px; background-color: #f0f9ff; border: 1px solid #bae6fd;">'
                      f'<div style="font-weight: 600; color: #0284c7; margin-bottom: 5px;">This Week</div>'
                      f'<div style="color: #334155; font-size: 0.9rem;">Focus on your top recommendation and track your progress daily.</div>'
                      f'</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown(f'<div style="padding: 12px; border-radius: 8px; background-color: #f0fdf4; border: 1px solid #bbf7d0;">'
                      f'<div style="font-weight: 600; color: #16a34a; margin-bottom: 5px;">This Month</div>'
                      f'<div style="color: #334155; font-size: 0.9rem;">Implement at least three recommendations and schedule a follow-up assessment.</div>'
                      f'</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
       
with tabs[2]:
    st.markdown('<h2 class="subheader">Model Insights</h2>', unsafe_allow_html=True)
    
    # Model information
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">About the Prediction Model</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="margin-bottom: 16px;">This application uses a machine learning model trained on health and lifestyle data to predict obesity risk categories.</p>
    
    <p style="margin-bottom: 16px;">The model analyzes 16 different factors including demographics, diet habits, physical activity, and lifestyle choices to classify individuals into one of seven weight categories.</p>
    
    <div style="padding: 16px; border-radius: 8px; background-color: #f8fafc; margin-bottom: 16px;">
        <p style="font-weight: 600; margin-bottom: 8px;">Weight Categories:</p>
        <ul style="margin-left: 20px; margin-bottom: 0;">
            <li><span style="color: #3b82f6; font-weight: 500;">Insufficient Weight</span> - BMI below 18.5</li>
            <li><span style="color: #10b981; font-weight: 500;">Normal Weight</span> - BMI between 18.5 and 24.9</li>
            <li><span style="color: #f59e0b; font-weight: 500;">Overweight Level I</span> - BMI between 25 and 27.49</li>
            <li><span style="color: #f97316; font-weight: 500;">Overweight Level II</span> - BMI between 27.5 and 29.9</li>
            <li><span style="color: #ef4444; font-weight: 500;">Obesity Type I</span> - BMI between 30 and 34.9</li>
            <li><span style="color: #dc2626; font-weight: 500;">Obesity Type II</span> - BMI between 35 and 39.9</li>
            <li><span style="color: #b91c1c; font-weight: 500;">Obesity Type III</span> - BMI 40 or higher</li>
        </ul>
    </div>
    
    <p>The model uses XGBoost, a powerful gradient boosting algorithm that excels at classification tasks. It has been trained on a dataset of individuals with various health profiles and validated to ensure accuracy.</p>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Important factors visualization
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Key Factors in Obesity Risk</h3>', unsafe_allow_html=True)
    
    # Check if model is available
    if model:
        try:
            sample_input = np.array([[0, 30, 170, 70, 0, 0, 1, 1, 1, 0, 1, 0, 1, 2, 1, 0]])
            
            explainer = create_shap_explainer(model)  # Ensure your function works correctly
            feature_names = get_feature_names()
            
            create_shap_bar_plot(explainer, sample_input, feature_names)  # Updated function call
            
        except Exception as e:
            st.error(f"Error generating SHAP insights: {e}")

    else:
        st.warning("Model not available. Unable to display feature importance.")
    
    # Display category encoding in a beautiful format
    st.markdown("""
        <div style="
            background-color: #f8fafc; 
            padding: 16px; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); 
            margin-top: 20px;">
            <h3 style="color: #2563eb; font-size: 20px; margin-bottom: 10px;">Obesity Risk Categories Encoding</h3>
            <ul style="list-style-type: none; padding-left: 0; font-size: 16px;">
                <li><strong style="color: #3b82f6;">0:</strong> Insufficient Weight</li>
                <li><strong style="color: #10b981;">1:</strong> Normal Weight</li>
                <li><strong style="color: #f59e0b;">2:</strong> Obesity Type I</li>
                <li><strong style="color: #f97316;">3:</strong> Obesity Type II</li>
                <li><strong style="color: #ef4444;">4:</strong> Obesity Type III</li>
                <li><strong style="color: #dc2626;">5:</strong> Overweight Level I</li>
                <li><strong style="color: #b91c1c;">6:</strong> Overweight Level II</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tips section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Understanding Your Results</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="margin-bottom: 12px;">This tool provides predictions based on statistical patterns, but each person's health journey is unique. Here are some tips for interpreting your results:</p>
    
    <ol style="margin-left: 16px;">
        <li><strong>BMI is just one indicator</strong> - While useful, BMI doesn't account for muscle mass, body composition, or individual health circumstances.</li>
        <li><strong>Focus on modifiable factors</strong> - Pay attention to lifestyle factors you can change, like diet, physical activity, and sleep habits.</li>
        <li><strong>Small changes add up</strong> - Even modest improvements in key areas can significantly reduce obesity risk over time.</li>
        <li><strong>Consult healthcare professionals</strong> - Use these insights as conversation starters with your doctor or nutritionist.</li>
    </ol>
    
    <div style="padding: 12px; border-radius: 8px; background-color: #f8fafc; margin-top: 16px; border-left: 4px solid #3b82f6;">
        <p style="margin: 0; font-style: italic;">This tool is for informational purposes and does not replace professional medical advice, diagnosis, or treatment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0;">
    <p style="color: #64748b; font-size: 0.9rem;">© 2025 Obesity Risk Assessment Tool | ECC Coding Week. All rights reserved</p>
</div>
""", unsafe_allow_html=True)