#!/usr/bin/python3

#import matplotlib.pyplot as plt
from ase.io import read, write
import numpy as np
import tol_colors as tc
cset = tc.bright

from test import KMeans



if __name__ == '__main__':
	atoms = read('SD22NQQ_wB97XD3BJ_Psi4.xyz', ':5')

	i = 0
	for a in atoms:
		km = KMeans()
		km.fit(a.positions,a.info['Nmols'], n_init=10, tol=0.005)
		a.arrays['molID'] = km.labels
		i += 1
	print(i)
	write('out.xyz',atoms)