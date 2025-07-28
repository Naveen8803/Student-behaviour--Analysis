import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========== Advanced CSS Styling ==========
st.markdown("""
    <style>
    body {
        background-color: #f4f6fa;
    }
    .main {
        background-color: #f4f6fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7, #182848);
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2c3e50;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ========== App Title ==========
st.title("🎓 Student Academic Performance Predictor")
st.markdown("Welcome! This app predicts student exam scores based on lifestyle and academic factors.")

# ========== Sidebar ==========
with st.sidebar:
    st.header("💡 Tips for Students")
    st.success("✅ 7–8 hours of sleep improves performance!")
    st.info("📘 This app uses machine learning models trained on real data.")
    st.markdown("Made with ❤️ using Streamlit.")

# ========== Load Models & Scaler ==========
@st.cache_resource
def load_model(model_name):
    with open(model_name, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

model_files = {
    "Linear Regression": "linear_model.pkl",
    "Ridge Regression": "ridge_model.pkl",
    "Lasso Regression": "lasso_model.pkl",
    "Random Forest": "random_forest_model.pkl"
}

model_choice = st.selectbox("📊 Choose a model", list(model_files.keys()))
model = load_model(model_files[model_choice])
scaler = None if model_choice == "Random Forest" else load_scaler()

# ========== Features ==========
feature_columns = [
    'age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
    'attendence_percentage', 'sleep_hours', 'exercise_frequency',
    'mental_health_rating', 'dq_e', 'ip_e', 'pel_e',
    'gender_Female', 'gender_Male',
    'part_time_job_No', 'part_time_job_Yes',
    'extracurricular_participation_No', 'extracurricular_participation_Yes'
]

# Label Encoding Maps
label_maps = {
    'gender': {'Female': 0, 'Male': 1},
    'part_time_job': {'No': 0, 'Yes': 1},
    'diet_quality': {'Poor': 0, 'Fair': 1, 'Good': 2},
    'parental_education_level': {'High School': 0, 'Bachelor': 1, 'Master': 2},
    'internet_quality': {'Poor': 0, 'Average': 1, 'Good': 2},
    'extracurricular_participation': {'No': 0, 'Yes': 1}
}

# ========== Input Form ==========
with st.form("input_form"):
    st.subheader("📥 Enter Student Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("🎂 Age", 10, 30, 18)
        gender = st.selectbox("🚻 Gender", ["Female", "Male"])
        parental_education = st.selectbox("🎓 Parental Education", ["High School", "Bachelor", "Master"])
        part_time_job = st.selectbox("💼 Part-time Job", ["No", "Yes"])
        extracurricular = st.selectbox("🏀 Extracurricular Activities", ["No", "Yes"])
    with col2:
        study_hours_per_day = st.slider("📚 Study Hours/Day", 0, 12, 4)
        social_media_hours = st.slider("📱 Social Media Hours/Day", 0.0, 4.0, 1.0)
        netflix_hours = st.slider("🎬 Netflix Hours/Day", 0.0, 4.0, 1.0)
        sleep_hours = st.slider("😴 Sleep Hours/Day", 0, 12, 7)
        attendence_percentage = st.slider("🏫 Attendance (%)", 0, 100, 75)

    col3, col4 = st.columns(2)
    with col3:
        diet_quality = st.selectbox("🍽️ Diet Quality", ["Poor", "Fair", "Good"])
        exercise_frequency = st.slider("🏋️‍♂️ Exercise Days/Week", 0, 7, 3)
    with col4:
        internet_quality = st.selectbox("🌐 Internet Quality", ["Poor", "Average", "Good"])
        mental_health_rating = st.slider("🧠 Mental Health (1-10)", 1, 10, 7)

    submitted = st.form_submit_button("🔍 Predict Score")

# ========== Input Encoding ==========
def encode_input():
    dq_e = label_maps['diet_quality'][diet_quality]
    ip_e = label_maps['internet_quality'][internet_quality]
    pel_e = label_maps['parental_education_level'][parental_education]
    gender_female = 1 if gender == "Female" else 0
    gender_male = 1 if gender == "Male" else 0
    part_time_no = 1 if part_time_job == "No" else 0
    part_time_yes = 1 if part_time_job == "Yes" else 0
    extracurricular_no = 1 if extracurricular == "No" else 0
    extracurricular_yes = 1 if extracurricular == "Yes" else 0

    input_dict = {
        'age': age,
        'study_hours_per_day': study_hours_per_day,
        'social_media_hours': social_media_hours,
        'netflix_hours': netflix_hours,
        'attendence_percentage': attendence_percentage,
        'sleep_hours': sleep_hours,
        'exercise_frequency': exercise_frequency,
        'mental_health_rating': mental_health_rating,
        'dq_e': dq_e,
        'ip_e': ip_e,
        'pel_e': pel_e,
        'gender_Female': gender_female,
        'gender_Male': gender_male,
        'part_time_job_No': part_time_no,
        'part_time_job_Yes': part_time_yes,
        'extracurricular_participation_No': extracurricular_no,
        'extracurricular_participation_Yes': extracurricular_yes
    }

    return np.array([input_dict[col] for col in feature_columns]).reshape(1, -1)

# ========== Prediction ==========
if submitted:
    try:
        input_data = encode_input()
        if scaler is not None:
            input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)[0]
        prediction = np.clip(prediction, 0, 100)

        st.success(f"🎯 Predicted Exam Score: **{prediction:.2f} / 100**")
        st.balloons()

        # ========== Charts ==========
        st.markdown("---")
        st.subheader("📊 Daily Time Allocation")
        time_labels = ['Study', 'Social Media', 'Netflix', 'Sleep', 'Others']
        time_values = [
            study_hours_per_day, social_media_hours,
            netflix_hours, sleep_hours,
            max(0, 24 - (study_hours_per_day + social_media_hours + netflix_hours + sleep_hours))
        ]
        fig1, ax1 = plt.subplots()
        ax1.pie(time_values, labels=time_labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        st.subheader("📈 Wellness Overview")
        wellness_data = {
            'Attendance (%)': attendence_percentage,
            'Exercise Days/Week': exercise_frequency,
            'Mental Health': mental_health_rating
        }
        st.bar_chart(pd.DataFrame(wellness_data, index=["Value"]))

        st.markdown("### 🧾 Input Summary")
        st.table(pd.DataFrame({
            "Feature": ["Gender", "Part-time Job", "Diet Quality", "Parental Education",
                        "Internet Quality", "Extracurricular"],
            "Value": [gender, part_time_job, diet_quality, parental_education,
                      internet_quality, extracurricular]
        }))

    except Exception as e:
        st.error(f"⚠ Something went wrong: {e}")
