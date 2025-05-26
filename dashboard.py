import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.manifold import TSNE

# Load models once
@st.cache_resource
def load_models():
    preprocessor = joblib.load("Models/preprocessor.pkl")
    pca = joblib.load("Models/pca.pkl")
    kmeans = joblib.load("Models/kmeans.pkl")
    rf_model = joblib.load("Models/rf_model.pkl")
    return preprocessor, pca, kmeans, rf_model

preprocessor, pca, kmeans, rf_model = load_models()

st.title("Interactive Customer Segmentation & Prediction Dashboard")

# Upload data
uploaded_file = st.file_uploader("Upload Customer Data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")

    # Preprocess input (drop target if exists)
    df_input = df.drop(columns=['y'], errors='ignore')
    X_processed = preprocessor.transform(df_input)

    # Clustering & Dimensionality Reduction
    X_pca = pca.transform(X_processed)
    kmeans_labels = kmeans.predict(X_pca)
    df['KMeans_Cluster'] = kmeans_labels

    # Classification Prediction (if 'y' missing)
    if 'y' not in df.columns:
        y_pred = rf_model.predict(X_processed)
        df['Predicted_Response'] = y_pred

    # Sidebar Filters
    st.sidebar.header("Filter Options")
    clusters = sorted(df['KMeans_Cluster'].unique())
    selected_clusters = st.sidebar.multiselect("Select Clusters to Display", clusters, default=clusters)

    # Filter dataframe based on selection
    filtered_df = df[df['KMeans_Cluster'].isin(selected_clusters)]

    st.subheader("Filtered Data Preview")
    st.dataframe(filtered_df.head(50))

    # Cluster counts bar chart
    st.subheader("Customer Count per Cluster")
    cluster_counts = filtered_df['KMeans_Cluster'].value_counts().sort_index()
    fig, ax = plt.subplots()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='Set2', ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Count per Cluster")
    st.pyplot(fig)

    # Heatmap of Feature Means per Cluster
    st.subheader("Cluster Feature Means Heatmap")
    # Use numeric columns only (skip cluster and prediction columns)
    num_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns.drop(['KMeans_Cluster'], errors='ignore').tolist()
    cluster_means = filtered_df.groupby('KMeans_Cluster')[num_cols].mean()

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.heatmap(cluster_means.T, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5, ax=ax2)
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Features")
    st.pyplot(fig2)

    # PCA scatter plot colored by cluster
    st.subheader("PCA Visualization")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='Set2', ax=ax3, alpha=0.7)
    ax3.set_xlabel("Principal Component 1")
    ax3.set_ylabel("Principal Component 2")
    ax3.set_title("Customer Segments - PCA Projection")
    ax3.legend(title='Cluster')
    st.pyplot(fig3)

    # t-SNE Visualization toggle
    st.subheader("t-SNE Visualization")
    if st.button("Run t-SNE (takes a few seconds)"):
        tsne = TSNE(n_components=2, random_state=0, n_iter=1000)
        X_tsne = tsne.fit_transform(X_processed)

        fig4, ax4 = plt.subplots()
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=kmeans_labels, palette='Set2', ax=ax4, alpha=0.7)
        ax4.set_xlabel("t-SNE Component 1")
        ax4.set_ylabel("t-SNE Component 2")
        ax4.set_title("Customer Segments - t-SNE Projection")
        ax4.legend(title='Cluster')
        st.pyplot(fig4)

    # Download filtered dataset
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Clustered Data as CSV",
        data=csv,
        file_name='filtered_clustered_customers.csv',
        mime='text/csv'
    )
else:
    st.info("Please upload a CSV file to start.")
