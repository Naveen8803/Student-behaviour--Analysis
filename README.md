# Student Academic Performance Predictor
This project is a Streamlit-based web app that predicts a student's exam performance based on lifestyle and academic behavior data using multiple machine learning models.

📁 Project Structure

├── app.py                        # Streamlit web application

├── project.ipynb                # Jupyter Notebook for data analysis and model training

├── student_habits_performance.csv  # Dataset used for training

├── linear_model.pkl             # Trained Linear Regression model

├── ridge_model.pkl              # Trained Ridge Regression model

├── lasso_model.pkl              # Trained Lasso Regression model

├── random_forest_model.pkl      # Trained Random Forest model

├── scaler.pkl                   # Scaler used for linear-based models

└── README.md                    # This file


🧠 Features

Input student attributes like study time, sleep, social media usage, diet, etc.

Choose from 4 trained models:

Linear Regression

Ridge Regression

Lasso Regression

Random Forest

Visualizations:

Time distribution (pie chart)

Wellness overview (bar chart)

Display prediction with confidence-enhancing visuals.


📊 Dataset

Stored in student_habits_performance.csv, includes:

Demographic info (age, gender, parental education)

Lifestyle habits (study, sleep, social media)

Health and well-being factors

Exam performance (target variable)


🧪 Models Used

Trained on the dataset using project.ipynb. The following were saved as .pkl files:

Linear, Ridge, and Lasso Regression (standardized using a scaler)

Random Forest (no scaler needed)
