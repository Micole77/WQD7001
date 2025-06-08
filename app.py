import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from streamlit_extras.stylable_container import stylable_container
from typing import Dict, List, Tuple
import warnings
import numpy as np
import sklearn

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Attack Risk Assessment", 
    page_icon="❤️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #fef2f2 0%, #fce7f3 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .step-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .progress-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 10px;
    }
    
    .step-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin: 0 10px;
    }
    
    .step-active {
        background-color: #dc2626;
        color: white;
        border: 2px solid #dc2626;
    }
    
    .step-completed {
        background-color: #16a34a;
        color: white;
        border: 2px solid #16a34a;
    }
    
    .step-inactive {
        background-color: white;
        color: #9ca3af;
        border: 2px solid #d1d5db;
    }
    
    .stress-scale {
        display: grid;
        grid-template-columns: repeat(10, 1fr);
        gap: 0.25rem;
        margin: 1rem 0;
    }
    
    .stress-button {
        height: 3rem;
        border-radius: 0.375rem;
        border: none;
        color: white;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .risk-result {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .risk-high {
        background-color: #fef2f2;
        border: 2px solid #fca5a5;
        color: #dc2626;
    }
    
    .risk-medium {
        background-color: #fffbeb;
        border: 2px solid #fcd34d;
        color: #f59e0b;
    }
    
    .risk-low {
        background-color: #f0fdf4;
        border: 2px solid #86efac;
        color: #16a34a;
    }
    
    .recommendation-item {
        display: flex;
        align-items: flex-start;
        margin: 0.5rem 0;
        padding: 0.5rem;
        background: #f8fafc;
        border-radius: 8px;
    }
    
    .disclaimer {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Initialize session state
# --------------------------------------------------
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# --------------------------------------------------
# Step definitions
# --------------------------------------------------
STEPS = [
    {
        "title": "Personal Information",
        "icon": "👤",
        "fields": ['age', 'sex', 'bmi', 'income', 'country']
    },
    {
        "title": "Medical History", 
        "icon": "🩺",
        "fields": ['diabetes', 'family_history', 'previous_heart_problems', 'medication_use']
    },
    {
        "title": "Vital Signs",
        "icon": "📊", 
        "fields": ['heart_rate', 'systolic_pressure', 'diastolic_pressure', 'cholesterol', 'triglycerides']
    },
    {
        "title": "Lifestyle Factors",
        "icon": "🍎",
        "fields": ['smoking', 'obesity', 'alcohol_consumption', 'diet', 'stress_level']
    },
    {
        "title": "Activity & Sleep",
        "icon": "💪",
        "fields": ['exercise_hours_per_week', 'physical_activity_days_per_week', 'sedentary_hours_per_day', 'sleep_hours_per_day']
    }
]

# Field configurations
COUNTRIES = [
    'Argentina','Australia','Brazil','Canada','China','Colombia','France',
    'Germany','India','Italy','Japan','New Zealand','Nigeria','South Africa',
    'South Korea','Spain','Thailand','United Kingdom','United States','Vietnam'
]
COUNTRY_ENCODING = {country: idx for idx, country in enumerate(COUNTRIES)}

FIELD_CONFIGS = {
    'age': {'type': 'number', 'label': 'Age (years)', 'min_val': 0, 'max_val': 100, 'default': 50},
    'sex': {'type': 'select', 'label': 'Sex', 'options': ['Male', 'Female'], 'values': [1, 0]},
    'bmi': {'type': 'number', 'label': 'BMI (kg/m²)', 'min_val': 15.0, 'max_val': 50.0, 'default': 24.0, 'step': 0.1},
    'diabetes': {'type': 'select', 'label': 'Diabetes', 'options': ['No', 'Yes'], 'values': [0, 1]},
    'family_history': {'type': 'select', 'label': 'Family History of Heart Disease', 'options': ['No', 'Yes'], 'values': [0, 1]},
    'previous_heart_problems': {'type': 'select', 'label': 'Previous Heart Problems', 'options': ['No', 'Yes'], 'values': [0, 1]},
    'medication_use': {'type': 'select', 'label': 'Currently Taking Medication', 'options': ['No', 'Yes'], 'values': [0, 1]},
    'heart_rate': {'type': 'number', 'label': 'Heart Rate (bpm)', 'min_val': 30, 'max_val': 200, 'default': 80},
    'systolic_pressure': {'type': 'number', 'label': 'Systolic Blood Pressure (mmHg)', 'min_val': 70, 'max_val': 250, 'default': 120},
    'diastolic_pressure': {'type': 'number', 'label': 'Diastolic Blood Pressure (mmHg)', 'min_val': 40, 'max_val': 150, 'default': 80},
    'cholesterol': {'type': 'number', 'label': 'Cholesterol (mg/dL)', 'min_val': 0, 'max_val': 400, 'default': 200},
    'triglycerides': {'type': 'number', 'label': 'Triglycerides (mg/dL)', 'min_val': 50, 'max_val': 600, 'default': 150},
    'smoking': {'type': 'select', 'label': 'Smoking Status', 'options': ['Non-smoker', 'Smoker'], 'values': [0, 1]},
    'obesity': {'type': 'select', 'label': 'Obesity Status', 'options': ['Not obese', 'Obese'], 'values': [0, 1]},
    'alcohol_consumption': {'type': 'select', 'label': 'Alcohol Consumption', 'options': ['Non-drinker', 'Drinker'], 'values': [0, 1]},
    'diet': {'type': 'select', 'label': 'Diet Quality', 'options': ['Unhealthy', 'Average', 'Healthy'], 'values': [0, 1, 2]},
    'stress_level': {'type': 'scale', 'label': 'Stress Level (1-10)', 'min_val': 1, 'max_val': 10, 'default': 5},
    'exercise_hours_per_week': {'type': 'number', 'label': 'Exercise Hours per Week', 'min_val': 0.0, 'max_val': 50.0, 'default': 3.0, 'step': 0.5},
    'physical_activity_days_per_week': {'type': 'number', 'label': 'Physical Activity Days per Week', 'min_val': 0.0, 'max_val': 7.0, 'default': 3.0, 'step': 0.5},
    'sedentary_hours_per_day': {'type': 'number', 'label': 'Sedentary Hours per Day', 'min_val': 0.0, 'max_val': 24.0, 'default': 8.0, 'step': 0.5},
    'sleep_hours_per_day': {'type': 'number', 'label': 'Sleep Hours per Day', 'min_val': 0, 'max_val': 12, 'default': 7},
    'income': {'type': 'number', 'label': 'Income (USD per year)', 'min_val': 20000, 'max_val': 300000, 'default': 150000, 'step': 1000},
    'country': {'type': 'select', 'label': 'Country of Residence', 'options': COUNTRIES, 'values': list(range(len(COUNTRIES))), 'default': 0}
}

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_xgboost_model(model_path: Path):
    """Load XGBoost model"""
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except:
        return None

def render_progress_bar(current_step: int, total_steps: int):
    """Render progress bar"""
    progress_html = '<div class="progress-bar">'
    
    for i in range(total_steps):
        step_num = i + 1
        if step_num < current_step:
            circle_class = "step-circle step-completed"
            icon = "✓"
        elif step_num == current_step:
            circle_class = "step-circle step-active"
            icon = str(step_num)
        else:
            circle_class = "step-circle step-inactive"
            icon = str(step_num)
        
        progress_html += f'<div class="{circle_class}">{icon}</div>'
        
        if i < total_steps - 1:
            line_class = "step-completed" if step_num < current_step else "step-inactive"
            progress_html += f'<div style="flex: 1; height: 2px; background-color: {"#16a34a" if line_class == "step-completed" else "#d1d5db"}; margin: 0 10px;"></div>'
    
    progress_html += '</div>'
    st.markdown(progress_html, unsafe_allow_html=True)
    
    # Step title
    step_info = STEPS[current_step - 1]
    st.markdown(f"<div style='text-align: center; margin: 1rem 0;'><span style='color: #6b7280;'>Step {current_step} of {total_steps}: {step_info['icon']} {step_info['title']}</span></div>", unsafe_allow_html=True)

def render_stress_scale(field_key: str, current_value: int = None):
    """Render interactive stress level scale with large, color-coded buttons that stay highlighted."""
    st.markdown(f"**{FIELD_CONFIGS[field_key]['label']}**")

    cols = st.columns(10)

    stress_colors = [
        "#16a34a", "#22c55e", "#84cc16", "#eab308", "#f59e0b",
        "#f97316", "#ef4444", "#dc2626", "#b91c1c", "#991b1b"
    ]

    stress_labels = [
        "Very Low", "Low", "Low-Med", "Medium", "Medium",
        "Med-High", "High", "High", "Very High", "Extreme"
    ]

    selected_level = current_value
    if selected_level is None:
        selected_level = st.session_state.form_data.get(field_key, 5)
    if not isinstance(selected_level, int) or not (1 <= selected_level <= 10):
        selected_level = 5

    for i in range(10):
        with cols[i]:
            level = i + 1
            color = stress_colors[i]
            label = stress_labels[i]

            is_selected = selected_level == level
            border_style = "3px solid white" if is_selected else "2px solid transparent"
            font_weight = "bold" if is_selected else "normal"

            with stylable_container(
                key=f"color_container_{field_key}_{level}",
                css_styles=f"""
                        button {{
                            background-color: {color};
                            color: white;
                            width: 100%;
                            height: 40px;
                            font-size: 20px;
                            font-weight: {font_weight};
                            border: {border_style};
                            border-radius: 8px;
                            cursor: pointer;
                            transition: border 0.2s ease;
                        }}
                        button:hover {{
                            background-color: white;
                            color: black;
                        }}
                    """
            ):
                if st.button(str(level), key=f"stress_{field_key}_{level}", help=f"Level {level}: {label}"):
                    st.session_state.form_data[field_key] = level
                    st.rerun()

    # Show current selection
    if selected_level:
        st.info(f"Selected Level: {selected_level}/10 - {stress_labels[selected_level-1]}")

    return selected_level

def render_field(field_key: str):
    """Render individual form field"""
    config = FIELD_CONFIGS[field_key]
    
    if config['type'] == 'number':
        # Use widget key as the source of truth
        widget_key = f"input_{field_key}"
        
        # Initialize widget value if not exists
        if widget_key not in st.session_state:
            st.session_state[widget_key] = st.session_state.form_data.get(field_key, config['default'])
        
        value = st.number_input(
            config['label'],
            min_value=config['min_val'],
            max_value=config['max_val'],
            step=config.get('step', 1),
            key=widget_key
        )
        
        # Store the widget value in form_data
        st.session_state.form_data[field_key] = value
        
    elif config['type'] == 'select':
        options = config['options']
        values = config['values']
        widget_key = f"select_{field_key}"
        
        # Find current option for initial value
        current_value = st.session_state.form_data.get(field_key)
        current_option = options[0]  # default to first option
        if current_value is not None:
            try:
                if field_key == 'country':
                    if isinstance(current_value, str):
                        current_index = options.index(current_value)
                        current_option = current_value
                    else:
                        current_index = current_value
                        current_option = options[current_index] if 0 <= current_index < len(options) else options[0]
                else:
                    current_index = values.index(current_value)
                    current_option = options[current_index]
            except ValueError:
                current_option = options[0]
        
        # Initialize widget value if not exists (store the option string, not index)
        if widget_key not in st.session_state:
            st.session_state[widget_key] = current_option
        
        selected_option = st.selectbox(
            config['label'],
            options,
            key=widget_key
        )
        
        # Map selection back to numeric value
        selected_index = options.index(selected_option)
        st.session_state.form_data[field_key] = values[selected_index]
        
    elif config['type'] == 'scale':
        field_key = "stress_level"
        if field_key not in st.session_state.form_data:
            st.session_state.form_data[field_key] = 5

        render_stress_scale(field_key)

def validate_current_step() -> Tuple[bool, List[str]]:
    """Validate current step fields"""
    current_step_fields = STEPS[st.session_state.current_step - 1]['fields']
    missing_fields = []
    
    for field in current_step_fields:
        if field not in st.session_state.form_data or st.session_state.form_data[field] is None:
            missing_fields.append(FIELD_CONFIGS[field]['label'])
    
    return len(missing_fields) == 0, missing_fields

def simulate_prediction():
    """Simulate heart attack risk prediction"""

    model_columns = [ 
        'age', 'sex', 'bmi', 'income', 'diabetes', 'family history',
        'previous heart problems', 'medication use', 'heart rate',
        'systolic pressure', 'diastolic pressure', 'cholesterol', 'triglycerides',
        'smoking', 'obesity', 'alcohol consumption', 'stress level',
        'exercise hours per week', 'physical activity days per week',
        'sedentary hours per day', 'sleep hours per day', 'diet_Average',
        'diet_Healthy', 'diet_Unhealthy', 'country_Argentina','country_Australia',
        'country_Brazil','country_Canada','country_China','country_Colombia',
        'country_France','country_Germany','country_India','country_Italy',
        'country_Japan','country_New Zealand','country_Nigeria','country_South Africa'
    ]
    #,'country_South Korea','country_Spain','country_Thailand','country_United Kingdom', 'country_United States','country_Vietnam'
    # Create a dictionary with all model columns initialized to 0
    data = {col: 0 for col in model_columns}
    
    # Get the list of countries from your FIELD_CONFIGS
    countries_list = FIELD_CONFIGS['country']['options']

    # Update the dictionary with user input from the form
    for field_key, value in st.session_state.form_data.items():
        if field_key == 'country':
            # Handle one-hot encoding for country
            selected_country_name = countries_list[value]
            # Construct the column name, e.g., 'country_United States'
            country_column_name = f"country_{selected_country_name}"
            if country_column_name in data:
                data[country_column_name] = 1
        elif field_key == 'diet':
            # Handle one-hot encoding for diet
            if value == 2: data['diet_Healthy'] = 1
            elif value == 1: data['diet_Average'] = 1
            else: data['diet_Unhealthy'] = 1
        else:
            # Map other field names to match model expectations
            # (You may need to adjust these based on your model's actual feature names)
            model_field = field_key.replace('_', ' ')
            if model_field in data:
                data[model_field] = value
            elif field_key in data: # For fields without spaces
                data[field_key] = value

    # Create a DataFrame with the correct columns and order
    df = pd.DataFrame([data])
    df = df[model_columns]

    # Load the actual model (XGBoost)
    model_path = Path("xgboost_model_new.pkl")
    model = load_xgboost_model(model_path)

    if model is None:
        raise FileNotFoundError(f"Could not load model at {model_path}")
    else: 
        # Run the prediction
        probability = model.predict_proba(df)[0][1]
        prediction = int(model.predict(df)[0])
        risk_score = probability * 100

    # Determine risk level
    if risk_score > 70:
        risk_level = 'High'
    elif risk_score > 40:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'
    
    return {
        'score': round(risk_score, 1),
        'level': risk_level,
        'prediction': prediction,
        'recommendations': get_recommendations(risk_level)
    }

def get_recommendations(risk_level: str) -> List[str]:
    """Get recommendations based on risk level"""
    recommendations = {
        'Low': [
            'Continue maintaining a healthy lifestyle',
            'Regular check-ups every 6-12 months', 
            'Keep up with current exercise routine',
            'Maintain healthy diet and stress management'
        ],
        'Medium': [
            'Consult with your doctor within 2-4 weeks',
            'Consider lifestyle modifications',
            'Monitor blood pressure regularly',
            'Increase physical activity if possible',
            'Focus on stress reduction techniques'
        ],
        'High': [
            'Seek immediate medical consultation',
            'Consider comprehensive cardiac evaluation',
            'Implement strict lifestyle changes',
            'Regular monitoring of vital signs',
            'Follow up with cardiologist'
        ]
    }
    return recommendations.get(risk_level, [])

def render_prediction_result():
    """Render prediction results"""
    result = st.session_state.prediction_result
    
    # Risk level styling
    risk_class = f"risk-{result['level'].lower()}"
    risk_emoji = "🔴" if result['level'] == 'High' else "🟡" if result['level'] == 'Medium' else "🟢"
    
    st.markdown(f"""
    <div class="risk-result {risk_class}">
        <h2 style="margin-bottom: 1rem;">{risk_emoji} Risk Assessment Complete</h2>
        <div style="font-size: 3rem; font-weight: bold; margin: 1rem 0;">{result['score']}%</div>
        <div style="font-size: 1.5rem; font-weight: 600;">{result['level']} Risk</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature-contribution plot
    st.header("🔍 Feature Contribution Plot")
    st.image(
        "feature-contribution plot.png",  
        caption="Feature-contribution Plot",      
        width=1300            
    )

    # Recommendations
    st.subheader("📋 Recommendations")
    for i, rec in enumerate(result['recommendations']):
        st.markdown(f"""
        <div class="recommendation-item">
            <span style="color: #16a34a; margin-right: 0.5rem;">✓</span>
            <span>{rec}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Disclaimer:</strong> This assessment is for educational purposes only and should not replace professional medical advice. Please consult with a healthcare provider for proper medical evaluation.
    </div>
    """, unsafe_allow_html=True)
    
    # Reset button
    if st.button("🔄 Start New Assessment", key="reset_assessment"):
        st.session_state.current_step = 1
        st.session_state.form_data = {}
        st.session_state.prediction_result = None
        st.rerun()

# --------------------------------------------------
# Main application
# --------------------------------------------------
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: #dc2626; font-size: 2.5rem; margin-bottom: 0.5rem;">❤️ Heart Attack Risk Assessment</h1>
        <p style="color: #6b7280; font-size: 1.1rem;">Complete the assessment to evaluate your cardiovascular risk</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show prediction result if available
    if st.session_state.prediction_result:
        render_prediction_result()
        return
    
    # Progress bar
    render_progress_bar(st.session_state.current_step, len(STEPS))
    
    # Form container
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    
    current_step_info = STEPS[st.session_state.current_step - 1]
    st.subheader(f"{current_step_info['icon']} {current_step_info['title']}")
    
    # Render fields for current step
    fields = current_step_info['fields']
    
    if len(fields) <= 3:
        # Single column for fewer fields
        for field in fields:
            render_field(field)
    else:
        # Two columns for more fields
        col1, col2 = st.columns(2)
        for i, field in enumerate(fields):
            with col1 if i % 2 == 0 else col2:
                render_field(field)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.session_state.current_step > 1:
            if st.button("⬅️ Previous", key="prev_btn"):
                st.session_state.current_step -= 1
                st.rerun()
    
    with col3:
        if st.session_state.current_step < len(STEPS):
            if st.button("Next ➡️", key="next_btn", type="primary"):
                is_valid, missing_fields = validate_current_step()
                if is_valid:
                    st.session_state.current_step += 1
                    st.rerun()
                else:
                    st.error(f"Please fill in the following fields: {', '.join(missing_fields)}")
        else:
            if st.button("🔍 Get Risk Assessment", key="submit_btn", type="primary"):
                is_valid, missing_fields = validate_current_step()
                if is_valid:
                    with st.spinner("Analyzing your risk factors..."):
                        st.session_state.prediction_result = simulate_prediction()
                    st.rerun()
                else:
                    st.error(f"Please fill in the following fields: {', '.join(missing_fields)}")

if __name__ == "__main__":
    main()
