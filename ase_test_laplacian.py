#!/usr/bin/python3

#import matplotlib.pyplot as plt
from ase.io import read, write
import numpy as np
import tol_colors as tc
cset = tc.Bright

from test import KMeans


def euclidean_distance_matrix(X):
	X2 = np.sum(X**2, axis=1)[:, None]
	D2 = X2 + X2.T - 2*X@X.T
	np.maximum(D2, 0, out=D2)
	return D2**0.5

def adjacency_matrix(X,cutoff=None):
	N, _ = X.shape
	dists = euclidean_distance_matrix(X)
	if cutoff is not None:
		dists[dists>cutoff] = 0
	dists[dists>0] = 1
	return dists

def degree_matrix(A):
	N, _ = A.shape
	D = np.diag([A[:,n].sum() for n in range(N)])
	return D

def laplacian_matrix(X,cutoff=None):
	dists = euclidean_distance_matrix(X)
	A = adjacency_matrix(dists,cutoff=cutoff)
	D = degree_matrix(A)
	return D-A

def num_clusters(X,cutoff=None):
	L = laplacian_matrix(X,cutoff=cutoff)
	num_clusters = np.linalg.eigvals(L)[0]
	return int(num_clusters)



if __name__ == '__main__':
	atoms = read('SD22NQQ1_wB97XD3BJ_Psi4.xyz', ':100')

	i = 0
	t = []
	for a in atoms:
		km = KMeans()
		#print(a.positions)
		nmols = num_clusters(a.positions,cutoff=3)
		t.append(nmols==a.info['Nmols'])
		#km.fit(a.positions,a.info['Nmols'], n_init=10, tol=0.005)
		#a.arrays['molID'] = km.labels
		i += 1
	#print(i)
	n_true = sum(t)
	print('accuracy=', n_true/len(t))
	#write('out.xyz',atoms)