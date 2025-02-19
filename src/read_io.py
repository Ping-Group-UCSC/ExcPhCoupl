#!/usr/bin/env python3
from __future__ import print_function, division
import sys
import numpy as np
#from numpy import linalg as LA
#import matplotlib.pyplot as plt
from netCDF4 import Dataset
import itertools

# Importing specific functions and parameters from user-defined modules
from modules.param import *

from modules.func import *

def read_qpoints_yambo():
	"""Read q-points from Yambo and convert them to the appropriate coordinate system."""
	Q_yambo_rku = np.genfromtxt('./qpoints_yambo', skip_header=1)[:, 0:3]
	Q_yambo = np.einsum('ij,ni->nj', LA.inv(b_iku), Q_yambo_rku)
	print("Q_yambo:", Q_yambo)
	return Q_yambo

def read_bse_wavefunction(Q_yambo, A_exc, exc_freq):
	"""Read the finite Q BSE wavefunction from Yambo."""
	for fragment_ind in range(N_Q):
		f = Dataset(path_ex + f'ndb.BS_diago_Q{fragment_ind+1}', 'r')
		BS_EIGENSTATES = f.variables['BS_EIGENSTATES']
		BS_Energies = f.variables['BS_Energies']	
		iQ = k2ik(Q_yambo[fragment_ind])
		A_exc_Q = np.array(BS_EIGENSTATES)[0:beta, :, 0] + 1j * np.array(BS_EIGENSTATES)[0:beta, :, 1]
		A_exc[iQ] = A_exc_Q.reshape(beta, N_k, N_v, N_c)
		exc_freq[iQ] = BS_Energies[0:beta, 0]
	print('Exciton wavefunctions size:', A_exc.shape)
	print('Norm check:', np.abs(np.einsum('bkvc,bkvc->b', A_exc[0].conj(), A_exc[0])))
	return A_exc, exc_freq
	
def read_elph_data(Q_yambo, g_elph, ph_freq):
	"""Read the electron-phonon coupling data and phonon frequencies."""
	for fragment_ind in range(N_Q):
		f = Dataset(path_elph_data + str(fragment_ind + 1), 'r')
		ELPH_fragment = f.variables[f'ELPH_GKKP_Q{fragment_ind+1}']
		PH_FREQS_fragment = f.variables[f'PH_FREQS{fragment_ind+1}']
		
		arr_ELPH_fragment = np.array(ELPH_fragment)[:, N_e-N_v:N_e+N_c, N_e-N_v:N_e+N_c, :, :]
		arr_PH_FREQS_fragment = np.array(PH_FREQS_fragment)[0:nmodes]
		
		iQ = k2ik(Q_yambo[fragment_ind])
		arr_temp = arr_ELPH_fragment[:, :, :, :, 0] + 1j * arr_ELPH_fragment[:, :, :, :, 1]
		g_temp = np.array(arr_temp).reshape((N_k, N_c + N_v, N_c + N_v, nmodes))
		
		ph_freq[iQ] = arr_PH_FREQS_fragment
		for k_fragment_ind in range(N_k):
			ik = k2ik(Q_yambo[k_fragment_ind])
			g_elph[iQ, ik] = g_temp[k_fragment_ind]
			
	return g_elph, ph_freq
	
def normalize_elph(g_elph, ph_freq):
	"""Normalize the electron-phonon matrix with phonon frequencies."""
	ph_freq = np.sqrt(np.abs(ph_freq))
	print('Phonon frequencies @ Gamma (meV):', ph_freq[0] / meV)
	for q_ind, l_ind in itertools.product(range(N_Q), range(nmodes)):
		if np.abs(ph_freq[q_ind, l_ind]) > eps_acustic:
			g_elph[q_ind, :, :, :, l_ind] /= np.sqrt(2. * ph_freq[q_ind, l_ind])
		else:
			g_elph[q_ind, :, :, :, l_ind] = 0.
	print('Electron-phonon matrix size:', g_elph.shape)
	return g_elph, ph_freq
