# ====== SETUP (MUST BE FIRST STREAMLIT COMMAND) ======
import streamlit as st
st.set_page_config(
    page_title="✨ Diabetes AI Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== LOAD CUSTOM CSS ======
import os
import json # Added for loading local Lottie JSONs
import requests # Still needed for web requests if you keep load_lottie_from_url

def load_css(path):
    """Loads a CSS file and injects it into the Streamlit app."""
    try:
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found at {path}. Please ensure 'styles.css' is in the same directory.")

load_css("styles.css")

# ====== IMPORTS ======
import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
import time
import plotly.express as px
import plotly.graph_objects as go
import base64

# ====== BACKGROUND IMAGE ======
def get_base64_of_image(image_path):
    """Encodes an image to base64 for CSS embedding."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        # st.warning(f"Background image '{image_path}' not found. Ensure it's in the same directory as app.py.")
        return None # Return None if file not found
    except Exception as e:
        st.error(f"Error loading background image '{image_path}': {e}")
        return None

# Replace with your actual image path if different
background_image_path = "background.png"
background_base64 = get_base64_of_image(background_image_path)

# Inject background image CSS only if image is found and base64 encoded
if background_base64:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{background_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        /* 'lighten' or 'screen' blend modes often work well for light backgrounds */
        background-blend-mode: lighten;
        /* Fallback background color, matches light-bg from styles.css */
        background-color: var(--light-bg);
    }}
    </style>
    """, unsafe_allow_html=True)
else:
    # If no background image or error, ensure app still uses light background
    st.markdown("""
    <style>
    .stApp {
        background-color: var(--light-bg); /* Ensure light background even without image */
    }
    </style>
    """, unsafe_allow_html=True)


# ====== LOAD MODEL AND SCALER ======
@st.cache_resource
def load_models():
    """Loads the pre-trained model and scaler."""
    try:
        model = joblib.load("diabetes_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler files not found. Please ensure 'diabetes_model.pkl' and 'scaler.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

model, scaler = load_models()

# ====== LOTTIE ANIMATIONS (FIXED FOR 403 ERROR) ======

def load_lottie_local(filepath):
    """Loads a Lottie animation JSON from a local file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Lottie file not found at {filepath}. Please check the path and ensure the file exists.")
        return None
    except json.JSONDecodeError:
        st.warning(f"Error decoding JSON from {filepath}. The file might be corrupted.")
        return None
    except Exception as e:
        st.warning(f"An unexpected error occurred loading Lottie from {filepath}: {e}")
        return None

# IMPORTANT: Ensure these files exist in your 'lottie_assets' folder!
# Example paths assuming a 'lottie_assets' subfolder
#lottie_healthy = load_lottie_local("lottie_assets/healthy_anim.json")
#lottie_diabetic = load_lottie_local("lottie_assets/diabetic_anim.json")
#lottie_doctor = load_lottie_local("lottie_assets/doctor_anim.json")

# You can keep this function if you ever need to load from a URL again,
# but it's not used for the specific animations that were causing 403 errors.
def load_lottie_from_url(url):
    """Loads Lottie animation JSON from a URL."""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
        st.warning(f"Failed to load Lottie animation from {url} (Status: {r.status_code})")
        return None
    except requests.exceptions.Timeout:
        st.warning(f"Timeout while loading Lottie animation from {url}. Check your internet connection.")
        return None
    except requests.exceptions.RequestException as e:
        st.warning(f"Error loading Lottie animation from {url}: {e}")
        return None
    except Exception as e:
        st.warning(f"An unexpected error occurred loading Lottie from {url}: {e}")
        return None


# ====== SIDEBAR ======
with st.sidebar:
    # Theme toggle button and logic removed as per "light mode only" requirement

    st.markdown("""
    <div class="glass-panel">
        <h2>👨‍⚕️ Patient Health Metrics</h2>
        <p class="subtext">Adjust sliders & predict diabetes risk</p>
    </div>
    """, unsafe_allow_html=True)

    #if lottie_doctor:
     #   st_lottie(lottie_doctor, height=180, key="sidebar_anim")
    #else:
     #   st.info("Lottie 'Doctor' animation could not be loaded. Please check `lottie_assets/doctor_anim.json`.")

    with st.form("input_form"):
        pregnancies = st.slider("🤰 Pregnancies", 0, 20, 1)
        glucose = st.slider("🩸 Glucose (mg/dL)", 50, 300, 120)
        bp = st.slider("💓 BP (mm Hg)", 40, 130, 70)
        skin = st.slider("🖐️ Skin Thickness (mm)", 0, 100, 20)
        insulin = st.slider("💉 Insulin (μU/mL)", 0, 300, 80)
        bmi = st.slider("⚖️ BMI", 10.0, 60.0, 25.0)
        dpf = st.slider("🧬 DPF", 0.1, 2.5, 0.5, format="%.2f") # Added format for DPF
        age = st.slider("👴 Age", 10, 100, 33)

        submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)

# ====== MAIN UI ======
st.markdown("""
<div class="hero-section">
    <h1>AI-Powered <span class="gradient-text">Diabetes</span> Predictor</h1>
    <p class="hero-subtitle">Advanced machine learning for early health insights</p>
</div>
""", unsafe_allow_html=True)

# ====== PREDICTION LOGIC ======
if submitted:
    with st.spinner("🚀 Analyzing health data..."):
        time.sleep(1.5)  # Simulate processing

        # Create input dataframe
        input_df = pd.DataFrame({
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [bp],
            "SkinThickness": [skin],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [dpf],
            "Age": [age]
        })

        # Scale input features
        try:
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)
            prediction_proba = model.predict_proba(scaled_input)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.stop()

    # ====== RESULTS DISPLAY ======
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📊 Risk Probability")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba[0][1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Diabetes Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00d2ff"},
                'steps': [
                    {'range': [0, 30], 'color': "#2ecc71"}, # Green for low risk
                    {'range': [30, 70], 'color': "#f39c12"}, # Orange for medium risk
                    {'range': [70, 100], 'color': "#e74c3c"} # Red for high risk
                ],
            }
        ))
        # Ensure Plotly charts use transparent background
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # Key metrics display
        st.markdown("### 📈 Key Metrics")
        st.markdown(f"""
        <div class="metric-card">
            <p>Glucose: <strong>{glucose} mg/dL</strong></p>
        </div>
        <div class="metric-card">
            <p>BMI: <strong>{bmi:.1f}</strong></p>
        </div>
        <div class="metric-card">
            <p>Age: <strong>{age} years</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        risk_class = "high-risk" if prediction[0] == 1 else "low-risk"
        risk_message = "⚠️ High Risk Detected" if prediction[0] == 1 else "✅ Low Risk Detected"
        recommendation = "🩺 Consult a doctor immediately" if prediction[0] == 1 else "💪 Maintain healthy habits!"

        st.markdown(f"""
        <div class="result-card {risk_class}">
            <h2>{risk_message}</h2>
            <p>AI prediction confidence: <strong>{prediction_proba[0][1]*100:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        #if prediction[0] == 1 and lottie_diabetic:
         #   st_lottie(lottie_diabetic, height=200)
        #elif prediction[0] == 0 and lottie_healthy:
         #   st_lottie(lottie_healthy, height=200)
        #else:
         #   st.info("Lottie animation could not be loaded for the prediction result.")


        st.markdown(f"""
            <div class="metric-card">
                <p><strong>Recommendation:</strong> {recommendation}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

