import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

# Load dataset
df = pd.read_csv("Dataset/bank-additional-full.csv")

# Drop target for clustering
X_cluster = df.drop(columns=['y'])

# Preprocessing
num_cols = X_cluster.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_cluster.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols)
])

X_processed = preprocessor.fit_transform(X_cluster)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(X_pca)
df['KMeans_Cluster'] = kmeans_labels

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_pca)
df['DBSCAN_Cluster'] = dbscan_labels

# Save the preprocessor and clustering models
joblib.dump(preprocessor, "Models/preprocessor.pkl")
joblib.dump(kmeans, "Models/kmeans.pkl")
joblib.dump(dbscan, "Models/dbscan.pkl")
joblib.dump(pca, "Models/pca.pkl")

# Dimensionality Reduction Visualizations
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='Set2')
plt.title('K-Means Clustering - PCA Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# t-SNE for better visualization of customer segments
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X_processed)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=kmeans_labels, palette='Set2')
plt.title('t-SNE - Customer Segments')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Cluster')
plt.show()

# Model Evaluation for Clustering

# Silhouette Score for KMeans
sil_score_kmeans = silhouette_score(X_pca, kmeans_labels)
print(f"Silhouette Score for K-Means: {sil_score_kmeans:.2f}")

# Davies-Bouldin Index for KMeans
dbi_kmeans = davies_bouldin_score(X_pca, kmeans_labels)
print(f"Davies-Bouldin Index for K-Means: {dbi_kmeans:.2f}")

# WCSS (Within-Cluster Sum of Squares) for KMeans
wcss = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), wcss, marker='o')
plt.title('Elbow Method to Determine Optimal k (K-Means)')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Train Random Forest Classifier model
X = preprocessor.transform(df.drop(columns=['y', 'KMeans_Cluster', 'DBSCAN_Cluster']))
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

# Save classifier model
joblib.dump(clf, "Models/rf_model.pkl")

# Evaluate classifier
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the updated DataFrame
df.to_csv("Dataset/clustered_bank_data.csv", index=False)
