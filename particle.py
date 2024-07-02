import numpy as np


class Particle:
    """
    Represents a single particle in a PSO swarm for clustering.

    This class defines a particle's properties and behaviors relevant to
    the Particle Swarm Optimization (PSO) algorithm for clustering tasks.
    """

    def __init__(self, num_clusters, data, inertia=0.72, cognitive_coef=1.49, social_coef=1.49):

        self.num_clusters = num_clusters  # Number of clusters to discover
        self.centroids_pos = data[np.random.choice(
            # Randomly initialize centroids
            list(range(len(data))), self.num_clusters)]
        # Initial personal best value (set to infinity)
        self.personal_best_val = np.inf
        # Personal best centroid positions (copy of initial)
        self.personal_best_pos = self.centroids_pos.copy()
        # Velocity for centroid updates (initialized to zeros)
        self.velocity = np.zeros_like(self.centroids_pos)
        self.inertia = inertia  # Inertia weight for velocity update
        self.cognitive_coef = cognitive_coef  # Cognitive coefficient
        self.social_coef = social_coef  # Social coefficien
        # Stores the best clustering solution found by the particle (initially None)
        self.personal_best_clustering = None

    def update_personal_best(self, data: np.ndarray):
       
        distances = self._calculate_distances(
            data=data)  # Calculate distances between data points and centroids
        # Assign data points to clusters (based on minimum distances)
        clusters = np.argmin(distances, axis=0)
        clusters_ids = np.unique(clusters)  # Get unique cluster IDs

        # Ensure all clusters have data points assigned
        # (handle potential empty clusters during initialization)
        while len(clusters_ids) != self.num_clusters:
            deleted_clusters = np.where(
                np.isin(np.arange(self.num_clusters), clusters_ids) == False)[0]
            self.centroids_pos[deleted_clusters] = data[np.random.choice(
                list(range(len(data))), len(deleted_clusters))]
            distances = self._calculate_distances(data=data)
            clusters = np.argmin(distances, axis=0)
            clusters_ids = np.unique(clusters)

        new_val = self._fitness_function(
            # Calculate fitness value for current clustering
            clusters=clusters, distances=distances)
        if new_val < self.personal_best_val:  # Check if current solution is better than personal best
            self.personal_best_val = new_val  # Update personal best value
            # Update personal best centroid positions
            self.personal_best_pos = self.centroids_pos.copy()
            # Update best clustering solution found by the particle
            self.personal_best_clustering = clusters.copy()

    def update_velocity(self, global_best_pos: np.ndarray):

        r1 = np.random.rand()  # Random value between 0 and 1 for cognitive component
        r2 = np.random.rand()  # Random value between 0 and 1 for social component
        self.velocity = self.inertia * self.velocity +\
            self.cognitive_coef * r1 * (self.personal_best_pos - self.centroids_pos) +\
            self.social_coef * r2 * (global_best_pos - self.centroids_pos) # Update velocity

    def move_centroids(self, global_best_pos):
        """
        Updates particle centroids based on current velocity.
        """
        self.update_velocity(
            global_best_pos=global_best_pos)  # Update velocity first
        # Add velocity to current centroid positions
        new_pos = self.centroids_pos + self.velocity
        self.centroids_pos = new_pos.copy()  # Update centroids with the new positions

    def _calculate_distances(self, data: np.ndarray) -> np.ndarray:
        """
        Calculates distances between data points and all centroids.
        """
        distances = []
        for centroid in self.centroids_pos:
            # Calculate distances between each data point and the current centroid
            # Calculate Euclidean distances
            d = np.linalg.norm(data - centroid, axis=1)
            # Append the distances for this centroid to the distances list
            distances.append(d)
         # Convert the list of distances into a NumPy array for further calculations
        distances = np.array(distances)
        return distances

    def _fitness_function(self, clusters: np.ndarray, distances: np.ndarray) -> float:
        """
        Calculates fitness value based on current clustering solution.
        """

        total_distance = 0.0  # Initialize to accumulate distances across clusters
        num_clusters = len(set(clusters))  # Number of unique clusters
        for cluster_index in range(num_clusters):
            # Find the indices of data points belonging to the current cluster
            data_indices_in_cluster = np.where(clusters == cluster_index)[0]
            # Calculate the sum of distances between those data points and their assigned centroid
            if len(data_indices_in_cluster):
                sum_of_distances = sum(
                    distances[cluster_index][data_indices_in_cluster])
                # Calculate the average distance within the cluster by dividing the sum by the number of data points
                average_distance = sum_of_distances / \
                    len(data_indices_in_cluster)
                # Add this average distance to the total_distance
                total_distance += average_distance
        # Calculate the overall fitness by dividing the total_distance by the number of clusters
        fitness = total_distance / num_clusters
        return fitness  # Return the calculated fitness value
