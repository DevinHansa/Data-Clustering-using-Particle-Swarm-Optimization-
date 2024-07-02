import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score, \
    davies_bouldin_score, calinski_harabasz_score
from psoSwarm import PsoSwarm

# Set plot to True if you want to visualize the clustering results
plot = True

# Load the wine dataset
wine_data = pd.read_csv('wine_dataset.csv')

# Extract the target variable (Class) and features
clusters_true = wine_data['Class'].values
data_points = wine_data.drop(['Class'], axis=1).values

# If plot is True, consider only the first two features for visualization
if plot:
    data_points = data_points[:, :2]

# Initialize and run PSO clustering
pso = PsoSwarm(num_clusters=3, num_particles=20, data=data_points)
clusters_pred, gb_val = pso.start(iteration=1000, plot=plot)

# # Calculate ARI
ari = adjusted_rand_score(clusters_true, clusters_pred)
print('Adjusted Rand Index (ARI):', ari)

# # Calculate Silhouette Score
silhouette_avg = silhouette_score(data_points, clusters_pred)
print('Silhouette Score:', silhouette_avg)

# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(data_points, clusters_pred)
print("Davies-Bouldin Index:", db_index)

# Calculate Calinski-Harabasz Index
ch_index = calinski_harabasz_score(data_points, clusters_pred)
print("Calinski-Harabasz Index:", ch_index)

# For showing the actual clusters
# Mapping for converting class labels to numeric indices
mapping = {1: 0, 2: 1, 3: 2}
clusters_true = np.array([mapping[x] for x in clusters_true])
print('Actual classes = ', clusters_true)
