import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- 1. Data Loading & Caching ---
@st.cache_data
def load_data():
    # We use the 50/50 split for balanced training
    try:
        df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2021.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset not found! Please upload 'diabetes_binary_5050split_health_indicators_BRFSS2021.csv' to the app directory.")
        return None

# --- 2. Model Training ---
@st.cache_resource
def train_model(df):
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, X_test, y_test

# Initialize App
df = load_data()

if df is not None:
    model, accuracy, X_test, y_test = train_model(df)
    
    # --- Sidebar: User Inputs ---
    st.sidebar.header("📝 Enter Health Details")
    
    def user_input_features():
        # Demographics
        st.sidebar.subheader("Demographics")
        age = st.sidebar.slider("Age Category (1=18-24 ... 13=80+)", 1, 13, 5)
        sex = st.sidebar.radio("Sex", ["Female", "Male"])
        sex_val = 1 if sex == "Male" else 0
        
        education = st.sidebar.slider("Education Level (1-6)", 1, 6, 4)
        income = st.sidebar.slider("Income Level (1-8)", 1, 8, 5)
        
        # Health Vitals
        st.sidebar.subheader("Health Vitals")
        bmi = st.sidebar.slider("BMI", 12, 98, 25)
        high_bp = st.sidebar.checkbox("High Blood Pressure?")
        high_chol = st.sidebar.checkbox("High Cholesterol?")
        chol_check = st.sidebar.checkbox("Cholesterol Check in last 5 years?")
        stroke = st.sidebar.checkbox("History of Stroke?")
        heart_disease = st.sidebar.checkbox("Heart Disease or Attack?")
        
        # Lifestyle
        st.sidebar.subheader("Lifestyle")
        smoker = st.sidebar.checkbox("Smoked at least 100 cigarettes in life?")
        phys_activity = st.sidebar.checkbox("Physical Activity in past 30 days?")
        fruits = st.sidebar.checkbox("Consume Fruit 1+ times per day?")
        veggies = st.sidebar.checkbox("Consume Vegetables 1+ times per day?")
        hvy_alcohol = st.sidebar.checkbox("Heavy Alcohol Consumption?")
        
        # General Health
        st.sidebar.subheader("General Health")
        gen_hlth = st.sidebar.slider("General Health Rating (1=Excellent, 5=Poor)", 1, 5, 3)
        ment_hlth = st.sidebar.slider("Days of Poor Mental Health (past 30 days)", 0, 30, 0)
        phys_hlth = st.sidebar.slider("Days of Poor Physical Health (past 30 days)", 0, 30, 0)
        diff_walk = st.sidebar.checkbox("Difficulty Walking?")
        healthcare = st.sidebar.checkbox("Have Health Care Coverage?")
        no_doc_cost = st.sidebar.checkbox("Could not see doctor due to cost?")
        
        # Create DataFrame matching training columns exactly
        data = {
            'HighBP': int(high_bp),
            'HighChol': int(high_chol),
            'CholCheck': int(chol_check),
            'BMI': bmi,
            'Smoker': int(smoker),
            'Stroke': int(stroke),
            'HeartDiseaseorAttack': int(heart_disease),
            'PhysActivity': int(phys_activity),
            'Fruits': int(fruits),
            'Veggies': int(veggies),
            'HvyAlcoholConsump': int(hvy_alcohol),
            'AnyHealthcare': int(healthcare),
            'NoDocbcCost': int(no_doc_cost),
            'GenHlth': gen_hlth,
            'MentHlth': ment_hlth,
            'PhysHlth': phys_hlth,
            'DiffWalk': int(diff_walk),
            'Sex': sex_val,
            'Age': age,
            'Education': education,
            'Income': income
        }
        return pd.DataFrame([data])

    input_df = user_input_features()

    # --- Main Dashboard ---
    st.title("🩺 Diabetes Health Risk Predictor")
    st.markdown("This AI-powered tool assesses the risk of diabetes based on health indicators defined by the BRFSS.")

    # Tabs for Organization
    tab1, tab2, tab3 = st.tabs(["🔍 Prediction Engine", "📊 Data Insights", "⚙️ Model Performance"])

    # --- TAB 1: Prediction ---
    with tab1:
        st.subheader("Patient Risk Assessment")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Prediction Logic
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0][1] # Probability of class 1 (Diabetes)
            
            # Gauge Chart for Probability
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_proba * 100,
                title = {'text': "Diabetes Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff4b4b" if prediction_proba > 0.5 else "#00cc96"},
                    'steps': [
                        {'range': [0, 50], 'color': "#e6f9f0"},
                        {'range': [50, 100], 'color': "#ffe6e6"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.write("### Analysis Result")
            if prediction == 1:
                st.error("""
                **High Risk Detected**
                
                The model predicts a high likelihood of diabetes or pre-diabetes based on the provided indicators.
                
                **Recommended Actions:**
                * Consult a healthcare professional immediately.
                * Monitor blood sugar levels.
                * Review diet and physical activity.
                """)
            else:
                st.success("""
                **Low Risk Detected**
                
                The model predicts a low likelihood of diabetes.
                
                **Recommendations:**
                * Maintain a healthy lifestyle.
                * Continue regular check-ups.
                """)
            
            st.info(f"**Top Contributing Factors:** High BMI, General Health Rating, and Age are typically strong indicators in this model.")

    # --- TAB 2: Data Insights ---
    with tab2:
        st.subheader("Population Statistics (Training Data)")
        
        # Scatter plot: BMI vs GenHlth
        fig_scatter = px.box(df, x='Diabetes_binary', y='BMI', 
                             color='Diabetes_binary', 
                             labels={'Diabetes_binary': 'Diabetes Status (0=No, 1=Yes)'},
                             title="BMI Distribution by Diabetes Status")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
             # High BP count
            bp_counts = df.groupby(['Diabetes_binary', 'HighBP']).size().reset_index(name='Count')
            fig_bar = px.bar(bp_counts, x='Diabetes_binary', y='Count', color='HighBP', barmode='group',
                             title="High BP Prevalence vs Diabetes")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_b:
            # General Health Distribution
            fig_hist = px.histogram(df, x='GenHlth', color='Diabetes_binary', barmode='overlay',
                                    title="General Health Rating Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- TAB 3: Model Performance ---
    with tab3:
        st.subheader("Model Telemetry")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Model Accuracy", f"{accuracy:.2%}")
            st.markdown(f"**Training Set Size:** {len(df)} records")
            
        with m_col2:
            # Confusion Matrix
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['No Diabetes', 'Diabetes'], y=['No Diabetes', 'Diabetes'])
            fig_cm.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
            
        # Feature Importance
        importances = model.feature_importances_
        feature_names = X_test.columns
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(10)
        
        fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Top 10 Risk Factors (Feature Importance)")
        st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.warning("Awaiting data upload...")