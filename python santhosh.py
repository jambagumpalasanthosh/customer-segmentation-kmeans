# Import libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Sample dataset (replace with your dataset if available)
data = {
    'Annual_Income': [15, 16, 17, 18, 19, 60, 62, 64, 65, 66],
    'Spending_Score': [39, 81, 6, 77, 40, 50, 52, 55, 53, 54]
}

df = pd.DataFrame(data)

# Feature selection
X = df[['Annual_Income', 'Spending_Score']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Add cluster labels
df['Cluster'] = labels

# Evaluation Metrics
sil_score = silhouette_score(X_scaled, labels)
db_score = davies_bouldin_score(X_scaled, labels)

# Output
print("Clustered Data:\n", df)
print("\nSilhouette Score:", sil_score)
print("Davies-Bouldin Index:", db_score)