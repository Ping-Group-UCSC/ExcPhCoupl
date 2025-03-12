#!/usr/bin/env python3
from __future__ import print_function, division
import numpy as np
from numpy import linalg as LA
from netCDF4 import Dataset
import itertools
from logmod import log
from common.param import p
from common.constants import meV, eps_acustic

# Importing specific functions and parameters from user-defined modules
from common.param import *
from common.func import *

def read_qpoints_yambo():
	"""Read q-points from Yambo and convert them to the appropriate coordinate system."""
	qpt_fil = p.data_dir + '/qpoints_yambo'
	log.debug("\t qpt file: " + qpt_fil)
	Q_yambo_rku = np.genfromtxt(qpt_fil, skip_header=1)[:, 0:3]
	Q_yambo = np.einsum('ij,ni->nj', LA.inv(b_iku), Q_yambo_rku)
	log.info("Q_yambo: " + str(Q_yambo))
	return Q_yambo

def read_bse_wavefunction(Q_yambo, A_exc, exc_freq):
	"""Read the finite Q BSE wavefunction from Yambo."""
	log.debug("\t NQ: " + str(p.N_Q))
	for fragment_ind in range(p.N_Q):
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
	log.debug("\t elph data: " + p.path_elph_data)
	for fragment_ind in range(p.N_Q):
		f = Dataset(p.path_elph_data + str(fragment_ind + 1), 'r')
		ELPH_fragment = f.variables[f'ELPH_GKKP_Q{fragment_ind+1}']
		PH_FREQS_fragment = f.variables[f'PH_FREQS{fragment_ind+1}']
		
		arr_ELPH_fragment = np.array(ELPH_fragment)[:, p.N_e-p.N_v:p.N_e+p.N_c, p.N_e-p.N_v:p.N_e+p.N_c, :, :]
		arr_PH_FREQS_fragment = np.array(PH_FREQS_fragment)[0:p.nmodes]
		
		iQ = k2ik(Q_yambo[fragment_ind])
		log.debug("\t iQ: " + str(iQ) + " -> " + str(Q_yambo[fragment_ind]))
		arr_temp = arr_ELPH_fragment[:, :, :, :, 0] + 1j * arr_ELPH_fragment[:, :, :, :, 1]
		g_temp = np.array(arr_temp).reshape((p.N_k, p.N_c+p.N_v, p.N_c+p.N_v, p.nmodes))

		ph_freq[iQ] = arr_PH_FREQS_fragment
		for k_fragment_ind in range(p.N_k):
			ik = k2ik(Q_yambo[k_fragment_ind])
			log.debug("\t \t ik: " + str(ik) + " -> " + str(Q_yambo[k_fragment_ind]))
			g_elph[iQ, ik] = g_temp[k_fragment_ind]
			
	return g_elph, ph_freq
	
def normalize_elph(g_elph, ph_freq):
	"""Normalize the electron-phonon matrix with phonon frequencies."""
	ph_freq = np.sqrt(np.abs(ph_freq))
	log.info('\t Phonon frequencies @ Gamma (meV):' + str(ph_freq[0]/meV))
	for q_ind, l_ind in itertools.product(range(p.N_Q), range(nmodes)):
		if np.abs(ph_freq[q_ind, l_ind]) > eps_acustic:
			g_elph[q_ind, :, :, :, l_ind] /= np.sqrt(2. * ph_freq[q_ind, l_ind])
		else:
			log.warning("\t small phonon frequency: " + str(ph_freq[q_ind,l_ind]) + " < " + str(eps_acustic))
			g_elph[q_ind, :, :, :, l_ind] = 0.
	log.info('\t Electron-phonon matrix size: ' + str(g_elph.shape))
	return g_elph, ph_freq
