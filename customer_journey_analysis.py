#  Import libraries
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

#  Load the dataset
df = pd.read_csv("Dataset/bank-additional-full.csv")

#  Drop target for clustering
X_cluster = df.drop(columns=['y'])

#  Separate numeric and categorical columns
num_cols = X_cluster.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_cluster.select_dtypes(include=['object']).columns.tolist()

#  Define preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols)
])

#  Apply preprocessing
X_processed = preprocessor.fit_transform(X_cluster)

#  Dimensionality Reduction using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

#  KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans_labels = kmeans.fit_predict(X_pca)
df['KMeans_Cluster'] = kmeans_labels

#  DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_pca)
df['DBSCAN_Cluster'] = dbscan_labels

#  Save models
joblib.dump(preprocessor, "Models/preprocessor.pkl")
joblib.dump(pca, "Models/pca.pkl")
joblib.dump(kmeans, "Models/kmeans.pkl")
joblib.dump(dbscan, "Models/dbscan.pkl")

#  Clustering Evaluation
sil_score_kmeans = silhouette_score(X_pca, kmeans_labels)
dbi_kmeans = davies_bouldin_score(X_pca, kmeans_labels)

print(f"Silhouette Score for K-Means: {sil_score_kmeans:.2f}")
print(f"Davies-Bouldin Index for K-Means: {dbi_kmeans:.2f}")

#  Elbow Method for Optimal K
# wcss = []
# for k in range(2, 10):
#     km = KMeans(n_clusters=k, random_state=0)
#     km.fit(X_pca)
#     wcss.append(km.inertia_)

# plt.figure(figsize=(8, 5))
# plt.plot(range(2, 10), wcss, marker='o')
# plt.title("Elbow Method - Optimal K for KMeans")
# plt.xlabel("Number of clusters")
# plt.ylabel("WCSS")
# plt.grid(True)
# plt.show()

#  1. Heatmap of Cluster Feature Averages (KMeans)
numeric_data = df[num_cols + ['KMeans_Cluster']]
cluster_means = numeric_data.groupby('KMeans_Cluster').mean()

plt.figure(figsize=(12, 6))
sns.heatmap(cluster_means.T, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title("Cluster Feature Means (KMeans)")
plt.xlabel("Cluster")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

#  2. t-SNE + Pairplot Visualization
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X_processed)

# Add t-SNE results and cluster labels to DataFrame
tsne_df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
tsne_df['Cluster'] = kmeans_labels

sns.pairplot(tsne_df, hue='Cluster', palette='Set2', plot_kws={'alpha': 0.6})
plt.suptitle("Pairplot - t-SNE Components by Cluster", y=1.02)
plt.show()

#  Random Forest Classifier
X = preprocessor.transform(df.drop(columns=['y', 'KMeans_Cluster', 'DBSCAN_Cluster']))
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

# Save classifier
joblib.dump(clf, "Models/rf_model.pkl")

# Evaluate classifier
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save final dataset with cluster labels
df.to_csv("Dataset/clustered_bank_data.csv", index=False)
