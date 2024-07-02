from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score, \
    davies_bouldin_score, calinski_harabasz_score
from particle import Particle


class PsoSwarm:
    """
    Class implementing the PSO (Particle Swarm Optimization) algorithm for clustering.
    """

    def __init__(self, num_clusters: int, num_particles: int, data: np.ndarray, inertia=0.72, cognitive_coef=1.49, social_coef=1.49):
        self.num_clusters = num_clusters
        self.num_particles = num_particles
        self.data = data  # Data points to cluster
        self.particles = []  # List to store the particles in the swarm
        self.global_best_pos = None  # Position of the global best particle
        # Fitness value of the global best particle (initialize to positive infinity)
        self.global_best_val = np.inf
        # Clustering solution of the global best particle
        self.global_best_clustering = None
        self._generate_particles(
            inertia, cognitive_coef, social_coef)  # Generate particles

    def _print_initial(self, iteration, plot):  # Print initial swarm configuration
        print('Initialing swarm with', self.num_particles, 'Number of Particles, ', self.num_clusters, 'Clusters with', iteration,
              'Max Iterations and with Plot =', plot)
        print('Data=', self.data.shape[0], 'points in',
              self.data.shape[1], 'dimensions')

    def _generate_particles(self, inertia: float, cognitive_coef: float, social_coef: float):
        # Generate particles with random initial positions
        for i in range(self.num_particles):
            # Create a new particle with random initial position
            particle = Particle(num_clusters=self.num_clusters, data=self.data,
                                inertia=inertia, cognitive_coef=cognitive_coef, social_coef=social_coef)
            self.particles.append(particle)

    def update_global_best(self, particle):
        # Update global best if the particle's personal best is better
        if particle.personal_best_val < self.global_best_val:
            self.global_best_val = particle.personal_best_val  # Update global best value
            # Update global best position
            self.global_best_pos = particle.personal_best_pos.copy()
            # Update global best clustering solution
            self.global_best_clustering = particle.personal_best_clustering.copy()

    def start(self, iteration=1000, plot=False) -> Tuple[np.ndarray, float]:
        # Start the PSO algorithm
        # Print initial swarm configuration
        self._print_initial(iteration, plot)
        progress = []  # List to store the progress of the algorithm
        for i in range(iteration):
            #   Print progress every 200 iterations
            if i % 200 == 0:
                clusters = self.global_best_clustering
                print('iteration', i, 'GB =', self.global_best_val)
                print('best clusters so far = ', clusters)
                if plot:
                    # Plot the data points and centroids
                    centroids = self.global_best_pos
                    if clusters is not None:  # Plot the data points with clusters
                        plt.scatter(
                            self.data[:, 0], self.data[:, 1], c=clusters, cmap='plasma')
                        plt.scatter(
                            centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.7)
                        plt.show()
                    # if there is no clusters yet ( iteration = 0 ) plot the data with no clusters
                    else:
                        plt.scatter(self.data[:, 0], self.data[:, 1])
                        plt.show()

            for particle in self.particles:
                # Update personal best for each particle
                particle.update_personal_best(data=self.data)
                self.update_global_best(particle=particle)

            for particle in self.particles:
                # Move centroids for each particle
                self.move_centroids(particle, self.global_best_pos)
                progress.append(
                    [self.global_best_pos, self.global_best_clustering, self.global_best_val])

        print('Finished!')
        # Return the best clustering solution and its fitness value
        return self.global_best_clustering, self.global_best_val

    def move_centroids(self, particle, global_best_pos):
        # Move centroids for each particle
        # Update velocity first
        particle.update_velocity(global_best_pos=global_best_pos)
        # Move centroids based on velocity
        particle.move_centroids(global_best_pos=global_best_pos)

    def calculate_ari(self, true_labels) -> float:
        # Calculate Adjusted Rand Index (ARI) for the best clustering solution found
        if self.global_best_clustering is None:
            # If no clustering solution is found yet, return 0
            print("No clustering solution found yet.")
            return 0.0
        return adjusted_rand_score(true_labels, self.global_best_clustering)

    def calculate_silhouette_score(self) -> float:
        # Calculate Silhouette Score for the best clustering solution found
        if self.global_best_clustering is None:
            print("No clustering solution found yet.")
            return 0.0
        return silhouette_score(self.data, self.global_best_clustering)

    def calculate_davies_bouldin_index(self) -> float:
        # Calculate Davies-Bouldin Index for the best clustering solution found
        if self.global_best_clustering is None:
            print("No clustering solution found yet.")
            return 0.0
        return davies_bouldin_score(self.data, self.global_best_clustering)

    def calculate_calinski_harabasz_index(self) -> float:
        # Calculate Calinski-Harabasz Index for the best clustering solution found
        if self.global_best_clustering is None:
            print("No clustering solution found yet.")
            return 0.0
        return calinski_harabasz_score(self.data, self.global_best_clustering)
