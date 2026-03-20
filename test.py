#!/usr/bin/python3

import numpy as np


class KMeans:
	"""Simple implementation of K-Means clustering.

	This class performs unsupervised clustering using Euclidean distance,
	iterative centroid optimization, and sum of squared errors (SSE)
	as the convergence criterion.
	"""
	def __init__(self):
		self.n_iterations = 0
		self.SSE = None
		self.models_by_sse = {}


	def get_labels(self,X,centroids):
		"""Assign each sample to the nearest centroid.

		Computes the Euclidean distance from each data point to each centroid
		and assigns each point to the closest centroid.

		Parameters
		----------
		X : np.ndarray
			Input data of shape (n_samples, n_features).
		centroids : dict
			Dictionary mapping centroid indices to centroid vectors.

		Returns
		-------
		np.ndarray
			Array of shape (n_samples,) containing the assigned cluster label
			for each data point.
		"""
		# Compute Euclidean distance from each point to each centroid
		dists_to_centroids = np.array(
			[np.linalg.norm(X-centroid,axis=1) 
			for centroid in centroids.values()]
		)
		# Assign each point to the nearest centroid
		return np.argmin(dists_to_centroids,axis=0)
		

	def get_clusters(self, X):
		"""Assign each point to the nearest centroid.

		Parameters
		----------
		X : np.ndarray
			Array of shape (n_samples, n_features) representing data points.

		Returns
		-------
		dict
			Dictionary mapping centroid indices to arrays of assigned points.
		"""
		# Compute labels for each point
		labels = self.get_labels(X, self.centroids)
		
		# Sort indices by cluster assignment
		sorted_indices = labels.argsort()
		closest_centroid_ids = labels[sorted_indices]

		# Split sorted indices by cluster
		ascending_split_indices = [
			sorted_indices[closest_centroid_ids == e]
			for e in set(closest_centroid_ids)
		]

		# Build clusters dict mapping centroid keys to assigned points
		clusters = {
			centroid: X[ids, :]
			for centroid, ids in zip(self.centroids, ascending_split_indices)
		}
		return clusters


	def get_centroids(self,clusters):
		"""Compute new centroids as the mean of each cluster.

		Parameters
		----------
		clusters : dict
			Dictionary mapping cluster indices to arrays of points.

		Returns
		-------
		dict
			Dictionary mapping cluster indices to centroid vectors.
		"""
		# Compute the mean vector for each cluster
		centroids = {
			i:cluster.mean(axis=0)
			for i,cluster in clusters.items()
		}
		return centroids


	def get_sse(self,X,centroid):
		"""Compute the sum of squared errors for a cluster.

		Parameters
		----------
		X : np.ndarray
			Points belonging to a single cluster.
		centroid : np.ndarray
			Centroid of the cluster.

		Returns
		-------
		float
			Sum of squared errors for the cluster.
		"""
		# Compute squared difference from centroid
		SSE = (X-centroid)**2
		return SSE.sum()


	def optimize_centroids(self,X,tol,max_iter):
		"""Recursively optimize centroids until convergence.

		Optimization continues until either the SSE difference
		falls below the tolerance or the maximum iteration
		count is reached.

		Parameters
		----------
		X : np.ndarray
			Input data of shape (n_samples, n_features).
		k : int
			Number of clusters.
		tol : float
			Convergence threshold for SSE improvement.
		max_iter : int
			Maximum number of optimization iterations.
		"""
		# Assign points to the current centroids
		# Compute new centroids from current clusters
		# Compute labels corresponding to new centroids
		clusters = self.get_clusters(X)
		new_centroids = self.get_centroids(clusters)
		labels = self.get_labels(X,new_centroids)

		# Compute total SSE across all clusters
		new_SSE = sum([
			self.get_sse(cluster,centroid)
			for cluster,centroid in zip(
				clusters.values(),new_centroids.values()
			)
		])
		
		# Continue optimization if SSE improves beyond tolerance and max_iter not reached
		if (self.SSE is None) or (abs(new_SSE-self.SSE) > tol):
			self.centroids = new_centroids
			self.n_iterations += 1
			self.SSE = new_SSE
			# Recursively optimize centroids
			if self.n_iterations < max_iter:
				self.optimize_centroids(X=X,tol=tol,max_iter=max_iter)
		
		# Store this run if convergence criteria met or max_iter reached
		elif (self.SSE is not None or abs(new_SSE-self.SSE) > tol) and (self.n_iterations < max_iter):
			self.models_by_sse[self.SSE] = {
				'n_iterations':self.n_iterations,
				'clusters':clusters,
				'centroids':new_centroids,
				'labels':labels,
			}


	def fit(self,X,k,tol=0.05,n_init=10,max_iter=300):
		"""Fit the K-Means model using multiple random initializations.

		Parameters
		----------
		X : np.ndarray
			Input data of shape (n_samples, n_features).
		k : int
			Number of desired clusters.
		tol : float, optional
			Convergence threshold for SSE improvement (default is 0.05).
		n_init : int, optional
			Number of random initializations (default is 10).
		max_iter : int, optional
			Maximum optimization iterations per run (default is 300).

		Attributes
		----------
		SSE : float
			Final minimum sum of squared errors.
		n_iterations : int
			Number of iterations to convergence for best run.
		clusters : dict
			Final clustering assignment.
		centroids : dict
			Final centroids.
		"""
		n_samples, _ = X.shape
		for _ in range(n_init):
			# Randomly select split points for initial centroids
			random_indices = sorted(np.random.randint(1,n_samples,k-1))
			random_split = np.array_split(X,random_indices, axis=0)
			
			# Compute initial centroids from random splits
			self.centroids = {
				i:split.mean(axis=0)
				for i,split in enumerate(random_split)
			}
			
			# Optimize centroids recursively
			self.optimize_centroids(X=X,tol=tol,max_iter=max_iter)

		# Select the run with the minimum SSE
		self.SSE = min(self.models_by_sse)
		best = self.models_by_sse[self.SSE]
		self.n_iterations = best['n_iterations']
		self.clusters = best['clusters']
		self.centroids = best['centroids']
		self.labels = best['labels']
		self.num_unique = len(set(self.labels))


	def predict(self,X):
		"""Predict cluster labels for new data using the best fitted model.

		Uses the centroids corresponding to the minimum stored SSE from the
		fitted models to assign each input sample to a cluster.

		Parameters
		----------
		X : np.ndarray
			Input data of shape (n_samples, n_features).

		Returns
		-------
		np.ndarray
			Array of shape (n_samples,) containing predicted cluster labels.
		"""
		# Raise error if model has not been fitted
		if self.SSE is None:
			raise RuntimeError("Model must be fitted before calling predict().")
		best = self.models_by_sse[self.SSE]
		centroids = best['centroids']

		# Compute nearest-centroid labels for input data
		labels = self.get_labels(X,centroids)
		return labels


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import tol_colors as tc
	cset = tc.bright

	np.random.seed(17)

	N = 10000
	d = 2
	k = 20
	
	positions = np.random.uniform(0,1,(N,d))

	km = KMeans()
	#help(km)
	km.fit(positions,k,n_init=10,tol=0.05)

	clusters, means, SSE = km.clusters, km.centroids, km.SSE
	print(f'The optimal k-means fit with an SSE of {SSE:.4} took {km.n_iterations} iterations.')
	#print(km.labels)
	#print(km.predict(positions))


	fig, ax = plt.subplots()
	for key,values in clusters.items():
		x,y = values.T
		cx,cy = means[key]
		dists = np.linalg.norm(values-means[key],axis=1)
		
		ax.scatter(x,y,c=dists)
		ax.scatter(cx,cy,color='white',marker='x')
	plt.show()
