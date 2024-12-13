#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

# Streamlit app title
st.title("📊 iPhone Reviews Analysis and Prediction")

# Sidebar file upload
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])

# Sidebar to set random forest parameters
st.sidebar.header("⚙️ Model Parameters")
n_estimators = st.sidebar.slider("Number of Trees in Random Forest", 10, 500, 100)
test_size = st.sidebar.slider("Test Set Proportion", 0.1, 0.5, 0.2)

if uploaded_file is not None:
    # Load dataset
    review_ip = pd.read_csv(uploaded_file)
    st.write("### 📝 Dataset Preview")
    st.dataframe(review_ip.head())

    # Data Preprocessing
    review_ip['date'] = pd.to_datetime(review_ip['date'], dayfirst=True)
    review_ip['year'] = review_ip['date'].dt.year
    review_ip['month'] = review_ip['date'].dt.month

    yearly_count = review_ip.groupby('year')['reviewTitle'].count().reset_index()
    yearly_count.columns = ['Year', 'Review Count']

    st.write("### 📈 Yearly Count of Reviews")
    st.line_chart(yearly_count.set_index('Year'))

    # Prediction
    X = yearly_count[['Year']]
    y = yearly_count['Review Count']

    # Scaling Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # Train the Model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and scaler
    model_filename = "random_forest_model.pkl"
    scaler_filename = "scaler.pkl"

    with open(model_filename, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(scaler_filename, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    st.write("✅ Model and Scaler saved as `.pkl` files.")

    # Evaluate the Model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("### 📊 Model Evaluation")
    st.write(f"- Mean Squared Error: {mse}")
    st.write(f"- R-Squared: {r2}")

    # Prediction for the next year
    next_year = np.array([[yearly_count['Year'].max() + 1]])
    next_year_scaled = scaler.transform(next_year)
    predicted_review_count = model.predict(next_year_scaled)

    st.write(f"📅 Predicted review count for the year {next_year[0][0]}: **{predicted_review_count[0]:.2f}**")

    # Visualization
    st.write("### 📊 Review Count Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(yearly_count['Year'], yearly_count['Review Count'], color='skyblue')
    ax.set_title('Year-wise Count of Reviews', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Review Count', fontsize=12)
    st.pyplot(fig)

    # Load model functionality
    st.sidebar.write("### 📂 Load Saved Model")
    loaded_model_file = st.sidebar.file_uploader("Upload a `.pkl` model file", type=["pkl"])
    if loaded_model_file is not None:
        loaded_model = pickle.load(loaded_model_file)
        st.write("✅ Model Loaded Successfully!")
else:
    st.write("🚀 Upload a CSV file to begin.")

st.write("ℹ️ This app allows you to analyze and predict review trends for iPhone data.")

# In[1]:





