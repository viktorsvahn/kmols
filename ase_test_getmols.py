#!/usr/bin/python3

#import matplotlib.pyplot as plt
import copy
from ase.io import read, write
import numpy as np
import tol_colors as tc
cset = tc.Bright

from test import KMeans


def get_dists(a):
	###################
	all_dists = a.get_all_distances(mic=False)
	triu = np.triu_indices(len(all_dists),k=1)
	dists = all_dists[triu]
	#print(dists.shape)
	return dists




def euclidean_distance_matrix(X):
	#############################
	X2 = np.sum(X**2, axis=1)[:, None]
	D2 = X2 + X2.T - 2*X@X.T
	np.maximum(D2, 0, out=D2)
	return D2**0.5


def get_mols(atoms, IDs=None):
	if IDs is None:
		assert 'molID' in a.arrays
		IDs = a.arrays['molID']
	mols = [atoms[IDs==ID] for ID in set(IDs)]
	return mols


def get_mol_centroids(mols):
	if len(mols) == 1:
		return np.mean(mols[0].positions, axis=0)
	elif len(mols) > 1:
		return np.array(
			[np.mean(mol.positions, axis=0) for mol in mols]
		)



def get_mol_adjacency(pos, mols):
	if isinstance(pos, (list, tuple)):
		pos = np.array(pos)
	mol_positions = get_mol_centroids(mols)
	#print(mol_positions, pos)
	try:
		distances = np.linalg.norm(mol_positions-pos, axis=1)
	except:
		distances = np.linalg.norm(mol_positions-pos)
	#print(distances.argsort())
	return distances.argsort()


def silhouette_index(atoms, IDs=None):
	"""Silhouette score for molID assignment."""
	mols = get_mols(atoms, IDs=IDs)
	scores = []
	for pos in atoms.positions:
		self_mol = get_mol_adjacency(pos, mols)[0]
		ai = np.mean(
			[np.linalg.norm(pos-p) for p in mols[self_mol].positions]
		)
		try:
			nearest_mol = get_mol_adjacency(pos, mols)[1]
			bi = np.mean(
				[np.linalg.norm(pos-p) for p in mols[nearest_mol].positions]
			)
		except:
			bi=ai
		si = (bi-ai)/max(ai,bi)
		scores.append(si)
	return np.mean(scores)



def wcss(atoms, IDs=None):
	"""Within cluster sum of squares (WCSS) for molID assignment using the Calinski–Harabasz index."""
	mols = get_mols(atoms, IDs=IDs)
	WCSS = 0.0
	for mol in mols:
		mu = np.mean(mol.positions, axis=0)
		WCSS += np.sum((mol.positions - mu) ** 2)
	return WCSS


def bcss(atoms, IDs=None):
	"""Between cluster sum of squares (BCSS) for molID assignment using the Calinski–Harabasz index."""
	mols = get_mols(atoms, IDs=IDs)
	BCSS = sum([np.linalg.norm(np.mean(mol.positions, axis=0)-np.mean(atoms.positions, axis=0))**2*len(mol) for mol in mols])
	
	#edm = euclidean_distance_matrix(get_mol_centroids(mols))
	#triu = np.triu_indices(len(edm),k=1)
	#mol_dists = edm[triu]
	#BCSS = sum(mol_dists**2)
	return BCSS


def calinski_harabasz_index(atoms, k, IDs=None):
	n = len(atoms)
	BCSS = bcss(atoms, IDs=IDs)
	WCSS = wcss(atoms, IDs=IDs)
	return (BCSS/WCSS)*(k-1)/(n-k)







def opt(atoms, method=None, nk=None):
	if nk is None:
		nk = len(a)
	

	scores = {
		'silhouette':[],
		'calinski_harabasz':[],
	}

	for k in range(1,nk):
		km = KMeans()
		km.fit(atoms.positions,k, n_init=100, tol=0.001)
		#print(len(set(km.labels)))

		if isinstance(method, str):
			if method.lower() == 'silhouette':
				scores['silhouette'].append(silhouette_index(atoms, IDs=km.labels))
				del scores['calinski_harabasz']
			
			elif method.lower() == 'calinski_harabasz':
				scores['calinski_harabasz'].append(calinski_harabasz_index(atoms, k, IDs=km.labels))
				del scores['silhouette']

		elif method is None:
			scores['silhouette'].append(silhouette_index(atoms, IDs=km.labels))
			scores['calinski_harabasz'].append(calinski_harabasz_index(atoms, k, IDs=km.labels))


	Kmeans = {method:np.argmax(values)+1 for method, values in scores.items()}
	print(scores)
	return Kmeans


if __name__ == '__main__':
	atoms = read('SD22NQQ1_wB97XD3BJ_Psi4.xyz', ':2')

	i = 0
	s = []
	ch = []
	for a in atoms:
		
		#s = opt(a, 'silhouette', nk=1)
		#res = opt(a, nk=2)
		res = opt(a)
		
		true_n = a.info['Nmols']
		print(res, true_n)
		s.append(true_n==res['silhouette'])
		ch.append(true_n==res['calinski_harabasz'])
		
		#x = bcss(a)
		#get_dists(a)

		#print(a.positions)
		#nmols = num_clusters(a.positions,cutoff=3)
		#t.append(nmols==a.info['Nmols'])
		#km.fit(a.positions,a.info['Nmols'], n_init=10, tol=0.005)
		#a.arrays['molID'] = km.labels
		i += 1
	#print(i)
	print('silhouette accuracy =', sum(s)/(len(s)+1e-16))
	print('Galinski-Harabasz accuracy =', sum(ch)/(len(ch)+1e-16))
	#write('out.xyz',atoms)