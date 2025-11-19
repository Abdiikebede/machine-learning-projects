🏡 House Price Prediction — Machine Learning Project
<p align="center"> <img src="https://i.imgur.com/sNCG9t5.png" width="70%" alt="House Price Prediction Banner"/> </p> <p align="center"> <b>Advanced Regression Techniques | Feature Engineering | Model Deployment</b> </p>
🔥 Badges
<p align="center"> <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python"/> <img src="https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikitlearn"/> <img src="https://img.shields.io/badge/XGBoost-Boosting-orange"/> <img src="https://img.shields.io/badge/GradientBoostingRegressor-Best%20Model-brightgreen"/> <img src="https://img.shields.io/badge/Kaggle-House%20Prices-blue?logo=kaggle"/> <img src="https://img.shields.io/badge/Status-Active-success"/> </p>

📘 Project Overview

This project builds a powerful machine learning model to predict house prices based on real estate features.
We use the official Kaggle dataset:

🔗 Kaggle Competition:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

🎯 Final Model Accuracy (R² Score): 87.16%

📁 Project Structure
.
├── house_price_prediction.ipynb   # Main ML workflow
├── submission.csv                 # Kaggle-ready predictions
├── gbr.pkl                        # Saved GradientBoosting model
├── requirements.txt               # Project dependencies
├── app.py                         # Flask API (optional)
├── streamlit_app.py               # Streamlit UI (optional)
└── README.md


🧠 Key Features
✔️ Full EDA (plots, correlations, distributions)
✔️ Data cleaning + preprocessing
✔️ Feature engineering
✔️ Model comparison across 10+ algorithms
✔️ Final model: GradientBoostingRegressor
✔️ Easy-to-use deployment templates (Flask + Streamlit)
✔️ Saved model (gbr.pkl) for reuse

🛠️ Technologies Used


Python


NumPy


Pandas


Matplotlib


Seaborn


Scikit-learn


XGBoost


Streamlit


Flask



🔍 Data Processing Workflow
1. Data Loading & EDA


Histograms, heatmaps, boxplots


Missing value detection


Outlier inspection


2. Data Preprocessing


Missing value imputation


One-hot encoding


Normalization and scaling


3. Model Training
Tested models include:


Linear Regression


Random Forest


XGBoost


GradientBoosting


MLPRegressor


KNN


Decision Trees


SVR


4. Model Evaluation


Cross-validation


R² score


Error comparison


5. Prediction & Export
Predictions saved as:
submission.csv


⭐ Project Logo (Downloadable)
<p align="center">
  <img src="https://i.imgur.com/vJoBNjz.png" width="40%" alt="House Price Prediction Logo">
</p>
If you want a custom banner with your name, I can generate one too.

📄 requirements.txt
Below is your ready-to-use requirements.txt:
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
flask
streamlit
joblib


🚀 Model Deployment Guide
You get two options: Flask API or Streamlit Web App.

1️⃣ Deploy Using Flask (REST API)
📌 app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load("gbr.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([np.array(data)])
    return jsonify({"predicted_price": float(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)

▶️ Run the API
python app.py

📩 Example Request
POST /predict
{
  "features": [12, 1400, 3, 2, 1, ...]
}


2️⃣ Deploy Using Streamlit (UI Web App)
📌 streamlit_app.py
import streamlit as st
import joblib
import numpy as np

model = joblib.load("gbr.pkl")

st.title("🏡 House Price Prediction App")

feature_values = st.text_input("Enter features (comma separated):")

if st.button("Predict"):
    values = np.array(list(map(float, feature_values.split(","))))
    pred = model.predict([values])
    st.success(f"Estimated Price: ${pred[0]:,.2f}")

▶️ Run the Streamlit App
streamlit run streamlit_app.py


📬 Contact
For improvements, collaboration, or questions:
📧 Email: abdikebede17@gmail.com

