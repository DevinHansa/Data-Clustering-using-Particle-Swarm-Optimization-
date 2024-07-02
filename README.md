# Data Clustering using Particle Swarm Optimization (PSO)

## Overview

This project implements data clustering using the Particle Swarm Optimization (PSO) algorithm. The aim is to evaluate the PSO algorithm's performance in clustering a comprehensive dataset and compare its effectiveness with traditional clustering methods.

## Objective

To leverage PSO for effective data clustering, assessing its strengths and limitations through various clustering quality metrics.

## Implementation

### PSO Algorithm

The PSO-based clustering algorithm is implemented in Python. Key components and steps include:

- **Particle Class**: Defines the properties and behaviors of particles in the PSO swarm, including initialization, velocity updates, centroid movement, and fitness evaluation.
- **PsoSwarm Class**: Manages the swarm of particles, updates global best positions, and evaluates the overall clustering quality.

### Key Parameters

- **Population size**: 20 particles
- **Number of iterations**: 1000
- **Inertia weight**: 0.72
- **Cognitive coefficient**: 1.49
- **Social coefficient**: 1.49

### Dataset

The Wine dataset was used for clustering experimentation. The dataset was preprocessed and formatted for clustering, with the target variable (Class) extracted and the features normalized.

### Evaluation Metrics

To assess the clustering quality, the following metrics were implemented:

- **Adjusted Rand Index (ARI)**
- **Silhouette Score**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Index**

### Visualizations

Visualizations were included to present the clustering results clearly, such as:

- **Cluster Centroids**
- **Cluster Assignments**
- **Evaluation Metric Scores**

## Results

The PSO-based clustering algorithm demonstrated significant potential in clustering tasks. The results, compared with traditional clustering methods, highlighted the strengths and limitations of the PSO approach.


## Conclusion

This project successfully implemented a PSO-based clustering algorithm and evaluated its performance on a comprehensive dataset. The PSO approach showed notable advantages in specific areas over traditional methods.




