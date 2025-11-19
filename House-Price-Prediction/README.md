![House Price Prediction Banner](https://i.imgur.com/vJoBNjz.png)
<p align="center">
  <b>🏡 House Price Prediction — Advanced Regression Project</b><br>
  End-to-End Machine Learning Pipeline | Kaggle Competition Ready | Deployable Web App & API
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E?logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-1.7+-E65F5C?logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/Best%20Model-GradientBoostingRegressor-success"/>
  <img src="https://img.shields.io/badge/Kaggle-House%20Prices-20BEFF?logo=kaggle&logoColor=white"/>
  <img src="https://img.shields.io/badge/R%C2%B2%20Score-87.16%25-brightgreen"/>
  <img src="https://img.shields.io/badge/License-MIT-blue"/>
  <img src="https://img.shields.io/badge/Status-Active-success"/>
</p>

---

### 📊 Project Overview
A complete **end-to-end machine learning project** that predicts house prices using the famous **Ames Housing dataset** from Kaggle's "House Prices - Advanced Regression Techniques" competition.

**Goal**: Build a robust regression model that accurately predicts sale prices based on 80+ features (square footage, location, quality, year built, etc.).

🔗 **Kaggle Competition Link**:  
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

**Best Model**: `GradientBoostingRegressor` → **R² Score: 87.16%**

---

### ✨ Key Features
- Comprehensive **Exploratory Data Analysis (EDA)** with beautiful visualizations
- Smart **missing value imputation** & **outlier handling**
- Advanced **feature engineering** (new meaningful features created)
- Compared **10+ regression algorithms**
- Hyperparameter tuning with cross-validation
- Ready-to-use **Flask REST API** and **Streamlit web app** for deployment
- Saved final model (`gbr.pkl`) using Joblib

---

### 🧠 Models Tested & Compared
| Model                     | R² Score  | Rank |
|---------------------------|-----------|------|
| GradientBoostingRegressor | **87.16%** | 1st  |
| XGBoost                   | 86.8%     | 2nd  |
| Random Forest             | 85.4%     | 3rd  |
| MLP Regressor (Neural Net)| 83.1%     |      |
| Linear Regression         | 81.2%     |      |
| SVR, KNN, Decision Tree   | < 80%     |      |

---

### 🛠 Tech Stack
- **Language**: Python 3.10+
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost
- **Deployment**: Flask (API), Streamlit (Web App)
- **Model Persistence**: Joblib

---


---

### 🚀 Quick Start

#### 1. Clone the repository
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
streamlit run streamlit_app.py
python app.py

# streamlit_app.py
import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/gbr.pkl")

st.title("🏡 House Price Prediction App")
st.write("Enter comma-separated feature values to get instant price estimation!")

features = st.text_input("Features (comma-separated):", "80,9000,3,2,2008,1990,...")

if st.button("🔮 Predict Price"):
    try:
        input_data = np.array([float(x) for x in features.split(",")]).reshape(1, -1)
        pred = model.predict(input_data)[0]
        st.success(f"**Estimated House Price: ${pred:,.2f}** 💰")
    except:
        st.error("Please enter valid numeric values.")

numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
flask
streamlit
joblib


📬 Contact & Collaboration
Love this project? Want to contribute or hire for similar work?
📧 Email: abdikebede17@gmail.com
💼 LinkedIn: linkedin.com/in/yourprofile
🐱 GitHub: github.com/Abdiikebede
