import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Streamlit configuration
st.set_page_config(page_title="iPhone Reviews Analysis", layout="wide")

# Title and description
st.title("iPhone Reviews Analysis")
st.write("An interactive app to analyze and predict iPhone review data.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    # Load dataset
    review_ip = pd.read_csv(uploaded_file)

    # Preprocessing
    review_ip['date'] = pd.to_datetime(review_ip['date'], dayfirst=True)
    review_ip['year'] = review_ip['date'].dt.year
    review_ip['month'] = review_ip['date'].dt.month

    # Display data
    st.subheader("Dataset Preview")
    st.dataframe(review_ip.head())

    # Yearly review count
    yearly_count = review_ip.groupby('year')['reviewTitle'].count().reset_index()
    yearly_count.columns = ['Year', 'Review Count']

    # Line chart for yearly reviews
    st.subheader("Year-wise Count of Reviews")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(yearly_count['Year'], yearly_count['Review Count'], marker='^')
    ax.set_title("Year-wise Count of Reviews")
    ax.set_xlabel("Year")
    ax.set_ylabel("Review Count")
    ax.grid(True)
    st.pyplot(fig)

    # Unique product and variant counts
    st.subheader("Unique Counts")
    st.write(f"Number of unique products: {review_ip['productAsin'].nunique()}")
    st.write(f"Number of unique variants: {review_ip['variant'].nunique()}")

    # Top 10 product variants by review count
    product_counts = review_ip.groupby('productAsin')['reviewTitle'].count().reset_index()
    product_counts.columns = ['Product Variant', 'Review Count']
    top_products = product_counts.sort_values(by='Review Count', ascending=False).head(10)

    st.subheader("Top 10 Product Variants by Review Count")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top_products['Product Variant'], top_products['Review Count'])
    ax.set_title("Top 10 Product Variants by Review Count")
    ax.set_xlabel("Product Variant")
    ax.set_ylabel("Review Count")
    ax.set_xticklabels(top_products['Product Variant'], rotation=90)
    ax.grid(True)
    st.pyplot(fig)

    # Prediction
    st.subheader("Predict Review Count for Next Year")
    X = yearly_count[['Year']]
    y = yearly_count['Review Count']

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump({'model': model, 'scaler': scaler}, 'model_scaler.pkl')
    st.write("Model and scaler saved to 'model_scaler.pkl'")

    # Load saved model and scaler
    saved_objects = joblib.load('model_scaler.pkl')
    loaded_model = saved_objects['model']
    loaded_scaler = saved_objects['scaler']

    # Prediction for next year
    next_year = np.array([[yearly_count['Year'].max() + 1]])
    next_year_scaled = loaded_scaler.transform(next_year)
    predicted_review_count = loaded_model.predict(next_year_scaled)
    st.write(f"Predicted review count for the year {next_year[0][0]}: {predicted_review_count[0]:.2f}")

    # Model evaluation
    y_pred = loaded_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")

else:
    st.write("Please upload a CSV file to start the analysis.")
