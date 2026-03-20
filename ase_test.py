#!/usr/bin/python3

#import matplotlib.pyplot as plt
from ase.io import read, write
import numpy as np
import tol_colors as tc
cset = tc.bright

#from aseMolec import anaAtoms as aa


from test import KMeans

def get_com(atoms, IDs=None):

	if IDs is None:
		assert 'molID' in a.arrays
		IDs = a.arrays['molID']


	mols = [a[a.arrays['molID']==ID] for ID in set(IDs)]
	com = np.array([mol.get_center_of_mass() for mol in mols])
	norms = np.linalg.norm(com, axis=1)
	#print(com)
	#print(norms)
	return com#[norms.argsort()]


if __name__ == '__main__':
	#atoms = read('SD22NQQ1_wB97XD3BJ_Psi4.xyz', ':')
	atoms = read('npt_100conEC_313_short.xyz', ':')

	i = 0
	t = []
	
	#aa.find_molecs(atoms, fct=1.0)
	for a in atoms:
		Nmols = a.info['Nmols']
		km = KMeans()
		km.fit(a.positions,Nmols, n_init=300, tol=2)
		print(Nmols, km.num_unique)
		
	"""
		A = get_com(a)
		
		#print(a.info['Nmols'])
		B = get_com(a, km.labels)
		#print(abs(A-B))
		#print(all(a.arrays['molID'] == km.labels))
		#print(any(abs(A-B)>0))

		t.append((abs(A-B)>0).any())
		#t.append(all(a.arrays['molID'] == km.labels))
		#print(t[-1])
		#i += 1
	#print(i)
	#n_true = sum(t)
	n_false = sum(t)
	print('accuracy=', 1-n_false/len(t))
	"""

	#print('accuracy=', n_true/len(t))
	#write('npt_100conEC_313_short.xyz',atoms)