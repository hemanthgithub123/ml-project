import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def save_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def main():
    st.title("iPhone Reviews Analysis")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        review_ip = load_data(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(review_ip.head())

        # Convert string to datetime
        if 'date' in review_ip.columns:
            review_ip['date'] = pd.to_datetime(review_ip['date'], dayfirst=True, errors='coerce')
            review_ip['year'] = review_ip['date'].dt.year
            review_ip['month'] = review_ip['date'].dt.month

        # Select scaling
        scaler = MinMaxScaler()
        scaled_cols = st.multiselect("Select Columns to Scale", options=review_ip.select_dtypes(include=np.number).columns)

        if scaled_cols:
            review_ip[scaled_cols] = scaler.fit_transform(review_ip[scaled_cols])
            st.write("### Scaled Data")
            st.dataframe(review_ip[scaled_cols].head())

        # Save the processed data to a .pkl file
        save_pkl(review_ip, "processed_reviews.pkl")
        st.write("### Processed Data Saved as `processed_reviews.pkl`")

        # Year-wise count of reviews
        if 'reviewTitle' in review_ip.columns and 'year' in review_ip.columns:
            yearly_count = review_ip.groupby('year')['reviewTitle'].count().reset_index()
            yearly_count.columns = ['Year', 'Review Count']

            # Line chart
            st.write("### Year-wise Count of Reviews")
            fig, ax = plt.subplots()
            ax.plot(yearly_count['Year'], yearly_count['Review Count'], marker='^')
            ax.set_title('Year-wise Count of Reviews (Line Chart)')
            ax.set_xlabel('Year')
            ax.set_ylabel('Review Count')
            ax.grid(True)
            st.pyplot(fig)

        # Top 10 product variants
        if 'productAsin' in review_ip.columns and 'reviewTitle' in review_ip.columns:
            product_counts = review_ip.groupby('productAsin')['reviewTitle'].count().reset_index()
            product_counts.columns = ['Product Variant', 'Review Count']
            top_products = product_counts.sort_values(by='Review Count', ascending=False).head(10)

            st.write("### Top 10 Product Variants by Review Count")
            fig, ax = plt.subplots()
            ax.bar(top_products['Product Variant'], top_products['Review Count'])
            ax.set_title('Top 10 Product Variants by Review Count')
            ax.set_xlabel('Product Variant')
            ax.set_ylabel('Review Count')
            ax.set_xticklabels(top_products['Product Variant'], rotation=90)
            ax.grid(True)
            st.pyplot(fig)

        # Group by 'productAsin' and 'year' for review trends
        if 'productAsin' in review_ip.columns and 'year' in review_ip.columns and 'reviewTitle' in review_ip.columns:
            review_counts = review_ip.groupby(['productAsin', 'year'])['reviewTitle'].count().reset_index()
            review_counts = review_counts.pivot(index='year', columns='productAsin', values='reviewTitle')

            st.write("### Review Trends by Product Variant")
            for product in review_counts.columns:
                fig, ax = plt.subplots()
                ax.plot(review_counts.index, review_counts[product], marker='o')
                ax.set_title(f'Review Trends for {product}')
                ax.set_xlabel('Year')
                ax.set_ylabel('Review Count')
                ax.grid(True)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
