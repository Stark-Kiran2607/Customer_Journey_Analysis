import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN

# Load saved models
preprocessor = joblib.load("Models/preprocessor.pkl")
kmeans = joblib.load("Models/kmeans.pkl")
dbscan = joblib.load("Models/dbscan.pkl")
pca = joblib.load("Models/pca.pkl")
rf_model = joblib.load("Models/rf_model.pkl")

# Load the dataset
df = pd.read_csv("Dataset/bank-additional-full.csv")

# Preprocessing data
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Apply preprocessor
X_processed = preprocessor.transform(df.drop(columns=['y']))

# Apply PCA for 2D representation
X_pca = pca.transform(X_processed)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X_processed)

# Streamlit Dashboard
st.title("Customer Journey Analysis Dashboard")
st.write("Explore customer behavior through clustering and dimensionality reduction.")

# Sidebar for user inputs
st.sidebar.header("Clustering & Dimensionality Reduction Settings")
cluster_method = st.sidebar.selectbox("Select Clustering Method", ["K-Means", "DBSCAN"])
dim_method = st.sidebar.selectbox("Select Dimensionality Reduction Method", ["PCA", "t-SNE"])

# Apply selected clustering
if cluster_method == "K-Means":
    st.subheader("K-Means Clustering Visualization")
    kmeans_labels = kmeans.predict(X_pca)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='Set2')
    plt.title("K-Means Clustering - PCA Projection")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    st.pyplot(plt)

elif cluster_method == "DBSCAN":
    st.subheader("DBSCAN Clustering Visualization")
    dbscan_labels = dbscan.fit_predict(X_pca)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dbscan_labels, palette='Set2')
    plt.title("DBSCAN Clustering - PCA Projection")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    st.pyplot(plt)

# Visualization for Dimensionality Reduction
if dim_method == "PCA":
    st.subheader("PCA Visualization")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans.predict(X_pca), palette='Set2')
    plt.title("PCA - Customer Segments")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    st.pyplot(plt)

elif dim_method == "t-SNE":
    st.subheader("t-SNE Visualization")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=kmeans.predict(X_pca), palette='Set2')
    plt.title("t-SNE - Customer Segments")
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Cluster')
    st.pyplot(plt)

# Clustering Evaluation Metrics
st.sidebar.header("Clustering Evaluation Metrics")
if st.sidebar.button("Evaluate Clustering"):

    # Silhouette Score for K-Means
    sil_score_kmeans = silhouette_score(X_pca, kmeans_labels)
    st.write(f"Silhouette Score for K-Means: {sil_score_kmeans:.2f}")

    # Davies-Bouldin Index for K-Means
    dbi_kmeans = davies_bouldin_score(X_pca, kmeans_labels)
    st.write(f"Davies-Bouldin Index for K-Means: {dbi_kmeans:.2f}")

    # WCSS (Within-Cluster Sum of Squares) for KMeans
    wcss = []
    for k in range(2, 10):
        kmeans_temp = KMeans(n_clusters=k, random_state=0)
        kmeans_temp.fit(X_pca)
        wcss.append(kmeans_temp.inertia_)

    st.subheader('WCSS (Within-Cluster Sum of Squares) for K-Means')
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 10), wcss, marker='o')
    plt.title('Elbow Method to Determine Optimal k (K-Means)')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    st.pyplot(plt)

# Model Prediction
st.sidebar.header("Model Prediction")
if st.sidebar.button("Predict Customer Outcome"):

    # Sample a random customer
    sample = df.drop(columns=['y']).sample(1)
    X_sample = preprocessor.transform(sample)
    
    # Predict using the Random Forest model
    prediction = rf_model.predict(X_sample)
    st.write(f"Predicted Outcome for Sample Customer: {prediction[0]}")
    
    # Show customer data for the prediction
    st.write("Sample Customer Data:")
    st.write(sample)
