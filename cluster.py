import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the iris dataset
file_path = "C:/Users/himasree9711/OneDrive/Desktop/sparks/Iris.csv"
iris_data = pd.read_csv(file_path)

# Prepare the data by excluding the Id and Species columns
X = iris_data.drop(['Id', 'Species'], axis=1)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimum number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Determining Optimum Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Based on the elbow plot, the optimal number of clusters can be determined
# Let's assume it's 3 for the iris dataset and fit the KMeans model
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
iris_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Cluster', data=iris_data, palette='viridis', s=100, alpha=0.7)
plt.title('Clusters of Iris Dataset')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()