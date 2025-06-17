#!/usr/bin/env python3
from __future__ import print_function, division
import numpy as np
from numpy import linalg as LA
from netCDF4 import Dataset
import itertools
import os
from logmod import log
from common.param import p
from common.constants import meV, eps_acustic
from mpi_module import mpi, MPI_ROOT

from netCDF4 import Dataset
import os
from common.param import p
from common.param import b_iku
from common.func import *

comm = mpi.comm
rank = mpi.rank
size = mpi.size
ROOT = MPI_ROOT

def read_qpoints_yambo():
	"""
	Read q-points.
	If MPI size > 1: Rank ROOT reads qpoints_yambo, then broadcasts.
	All ranks then convert to the appropriate coordinate system.
	"""
	qpt_file_path = os.path.join(p.data_dir, "qpoints_yambo")

	if size > 1:
		if rank == ROOT:
			log.debug(f"\t qpt file (MPI ROOT reading): {qpt_file_path}")
			with open(qpt_file_path, "r") as f:
				lines = f.readlines()

			_Q_yambo_rku_root = np.zeros((p.N_Q, 3), dtype=float)
			for i in range(p.N_Q):
				parts = lines[i+1].split()
				_Q_yambo_rku_root[i,0] = float(parts[0])
				_Q_yambo_rku_root[i,1] = float(parts[1])
				_Q_yambo_rku_root[i,2] = float(parts[2])
		else:
			_Q_yambo_rku_root = None

		# Broadcast Q_yambo_rku from ROOT to all ranks
		Q_yambo_rku = comm.bcast(_Q_yambo_rku_root, root=ROOT)

	else:
		if rank == ROOT:
			log.debug(f"\t qpt file (Serial reading): {qpt_file_path}")

		Q_yambo_rku = np.genfromtxt(qpt_file_path, skip_header=1, usecols=(0,1,2), dtype=float)

	Q_yambo = np.einsum('ij,ni->nj', LA.inv(b_iku), Q_yambo_rku)

	if rank == ROOT:
		log.info(f"Q_yambo shape: {Q_yambo.shape}")
	return Q_yambo

def read_bse_wavefunction(Q_yambo):
	"""
	Read the finite Q BSE wavefunction.
	If MPI size > 1: Each rank reads a subset of BSE files, gathered on ROOT, then broadcast.
	If MPI size == 1: Single process reads all files.
	Uses p.path_bse_data for the directory.
	Returns A_exc (complex64) and exc_freq (float32).
	"""
	bse_save_dir = p.path_bse_data

	if size > 1:
		num_frags_total = p.N_Q
		frags_per_rank, remainder = divmod(num_frags_total, size)
		start_frag = rank * frags_per_rank + min(rank, remainder)
		end_frag = start_frag + frags_per_rank + (1 if rank < remainder else 0)

		if rank == ROOT:

			_A_exc_root = np.zeros((p.N_Q, p.beta, p.N_k, p.N_v, p.N_c), dtype=np.complex64)
			_exc_freq_root = np.zeros((p.N_Q, p.beta), dtype=np.float32)
		else:
			_A_exc_root = None
			_exc_freq_root = None

		local_list_A = []
		local_list_freq = []
		for iQ_frag_index in range(start_frag, end_frag):
			fname = f"ndb.BS_diago_Q{iQ_frag_index + 1}"
			file_path = os.path.join(bse_save_dir, fname)

			if rank == ROOT and not os.path.isfile(file_path):
				log.warning(f"BSE file not found (rank {rank} looking for ROOT): {file_path}")
			elif not os.path.isfile(file_path) and rank != ROOT :
				log.warning(f"BSE file not found (rank {rank}): {file_path}")

			try:
				ds = Dataset(file_path, "r")
				bs_eigenstates_data = ds.variables["BS_EIGENSTATES"][:]
				bs_energies_data = ds.variables["BS_Energies"][:]
				ds.close()

				complex_states = bs_eigenstates_data[:p.beta, :, 0] + 1j * bs_eigenstates_data[:p.beta, :, 1]

				A_slice = complex_states.reshape((p.beta, p.N_k, p.N_v, p.N_c)).astype(np.complex64)
				freq_slice = bs_energies_data[:p.beta, 0].astype(np.float32)

				local_list_A.append((iQ_frag_index, A_slice))
				local_list_freq.append((iQ_frag_index, freq_slice))
			except FileNotFoundError:
				if rank == ROOT:
					log.error(f"FATAL: BSE file {file_path} not found by rank {rank}. Cannot proceed.")
				pass


		gathered_A = comm.gather(local_list_A, root=ROOT)
		gathered_freq = comm.gather(local_list_freq, root=ROOT)

		if rank == ROOT:
			for rdata_A in gathered_A:
				for iQ_idx, A_s in rdata_A:
					_A_exc_root[iQ_idx] = A_s
			for rdata_freq in gathered_freq:
				for iQ_idx, f_s in rdata_freq:
					_exc_freq_root[iQ_idx] = f_s

		# Broadcast the assembled arrays back to all ranks
		A_exc = comm.bcast(_A_exc_root, root=ROOT)
		exc_freq = comm.bcast(_exc_freq_root, root=ROOT)

	else:
		A_exc = np.zeros((p.N_Q, p.beta, p.N_k, p.N_v, p.N_c), dtype='complex64')
		exc_freq = np.zeros((p.N_Q, p.beta), dtype='float32')
		if rank == ROOT:
			log.debug(f"Reading BSE wavefunctions serially. NQ: {p.N_Q}")

		for fragment_ind in range(p.N_Q):
			fname = f"ndb.BS_diago_Q{fragment_ind + 1}"
			file_path = os.path.join(bse_save_dir, fname)
			if not os.path.isfile(file_path):
				log.error(f"BSE file not found: {file_path}")
				continue

			ds = Dataset(file_path, 'r')
			bs_eigenstates_data = ds.variables['BS_EIGENSTATES'][:]
			bs_energies_data = ds.variables['BS_Energies'][:]
			ds.close()

			iQ = fragment_ind

			complex_states = bs_eigenstates_data[:p.beta, :, 0] + 1j * bs_eigenstates_data[:p.beta, :, 1]
			A_exc_Q_reshaped = complex_states.reshape(p.beta, p.N_k, p.N_v, p.N_c)

			A_exc[iQ] = A_exc_Q_reshaped.astype(np.complex64)
			exc_freq[iQ] = bs_energies_data[:p.beta, 0].astype(np.float32)

	if rank == ROOT:
		log.info(f'Exciton wavefunctions A_exc shape: {A_exc.shape}')
		if p.N_Q > 0 and A_exc.shape[0] > 0 :
			norm_check_val = np.abs(np.einsum('kvc,kvc->', A_exc[0,0].conj(), A_exc[0,0]))
			log.info(f'Norm check (A_exc[0,0]): {norm_check_val}')
	return A_exc, exc_freq

def read_elph_data(Q_yambo):
	"""
	Read electron-phonon coupling data (g_elph) and phonon frequencies (ph_freq).
	If MPI size > 1: Parallel read of el-ph fragments, dimensions from sample file on ROOT.
	If MPI size == 1: Serial read, dimensions from sample file.
	Uses p.elph_dir for directory.
	Returns g_elph (complex64) and ph_freq (float32).
	The g_elph matrix is ELPH_GKKP_Q (N_k, actual_bands, actual_bands, actual_nmodes, 2).
	We assume actual_bands corresponds to p.N_v + p.N_c.
	And actual_nmodes corresponds to p.nmodes.
	"""
	elph_save_dir = p.elph_dir
	_actual_bands_val = 0
	_actual_nmodes_val = 0

	if rank == ROOT:
		sample_fname = os.path.join(elph_save_dir, "ndb.elph_gkkp_expanded_fragment_1")
		if not os.path.isfile(sample_fname):
			log.error(f"Missing sample el-ph file for dimension determination: {sample_fname}")
			_actual_bands_val = p.N_v + p.N_c
			_actual_nmodes_val = p.nmodes
			log.warning(f"Using default dimensions: actual_bands={_actual_bands_val}, actual_nmodes={_actual_nmodes_val}")
		else:
			ds_sample = Dataset(sample_fname, "r")
			sample_elph_var = ds_sample.variables["ELPH_GKKP_Q1"]
			_actual_bands_val = sample_elph_var.shape[1]
			_actual_nmodes_val = sample_elph_var.shape[3]
			ds_sample.close()
			log.info(f"Dimensions from sample el-ph file: actual_bands={_actual_bands_val}, actual_nmodes={_actual_nmodes_val}")

			if _actual_bands_val != (p.N_v + p.N_c):
				log.warning(f"actual_bands from file ({_actual_bands_val}) differs from p.N_v+p.N_c ({p.N_v+p.N_c})")
			if _actual_nmodes_val != p.nmodes:
				log.warning(f"actual_nmodes from file ({_actual_nmodes_val}) differs from p.nmodes ({p.nmodes})")


	if size > 1:
		actual_bands = comm.bcast(_actual_bands_val if rank == ROOT else None, root=ROOT)
		actual_nmodes = comm.bcast(_actual_nmodes_val if rank == ROOT else None, root=ROOT)
	else:
		actual_bands = _actual_bands_val
		actual_nmodes = _actual_nmodes_val

	g_elph_shape = (p.N_Q, p.N_k, actual_bands, actual_bands, actual_nmodes)
	ph_freq_shape = (p.N_Q, actual_nmodes)

	if size > 1:
		num_frags_total = p.N_Q
		frags_per_rank, remainder = divmod(num_frags_total, size)
		start_frag = rank * frags_per_rank + min(rank, remainder)
		end_frag = start_frag + frags_per_rank + (1 if rank < remainder else 0)

		if rank == ROOT:
			_g_elph_root = np.zeros(g_elph_shape, dtype=np.complex64)
			_ph_freq_root = np.zeros(ph_freq_shape, dtype=np.float32)
		else:
			_g_elph_root = None
			_ph_freq_root = None

		local_g_list = []
		local_ph_list = []

		for iQ_frag_index in range(start_frag, end_frag):
			fname = f"ndb.elph_gkkp_expanded_fragment_{iQ_frag_index + 1}"
			file_path = os.path.join(elph_save_dir, fname)

			if not os.path.isfile(file_path):
				if rank == ROOT: log.warning(f"ELPH file not found (rank {rank}): {file_path}")
				continue

			ds = Dataset(file_path, "r")
			elph_data_frag = ds.variables[f"ELPH_GKKP_Q{iQ_frag_index + 1}"][:]
			ph_freqs_frag = ds.variables[f"PH_FREQS{iQ_frag_index + 1}"][:]
			ds.close()

			if elph_data_frag.shape[0] != p.N_k:
				log.error(f"Mismatch in N_k for ELPH fragment {iQ_frag_index+1}: file has {elph_data_frag.shape[0]}, p.N_k is {p.N_k}")

			num_k_to_process = min(elph_data_frag.shape[0], p.N_k)

			g_complex_frag = elph_data_frag[:num_k_to_process, ..., 0] + 1j * elph_data_frag[:num_k_to_process, ..., 1]


			for ik_in_frag in range(num_k_to_process):
				g_slice = g_complex_frag[ik_in_frag].astype(np.complex64)
				local_g_list.append((iQ_frag_index, ik_in_frag, g_slice))

			ph_slice = ph_freqs_frag[:actual_nmodes].astype(np.float32)
			local_ph_list.append((iQ_frag_index, ph_slice))

		gathered_g = comm.gather(local_g_list, root=ROOT)
		gathered_ph = comm.gather(local_ph_list, root=ROOT)

		if rank == ROOT:
			for rdata_g in gathered_g:
				for iQ_idx, ik_idx, g_s in rdata_g:
					_g_elph_root[iQ_idx, ik_idx] = g_s
			for rdata_ph in gathered_ph:
				for iQ_idx, ph_s in rdata_ph:
					_ph_freq_root[iQ_idx] = ph_s

		g_elph = comm.bcast(_g_elph_root, root=ROOT)
		ph_freq = comm.bcast(_ph_freq_root, root=ROOT)

	else:
		g_elph = np.zeros(g_elph_shape, dtype=np.complex64)
		ph_freq = np.zeros(ph_freq_shape, dtype=np.float32)

		if rank == ROOT:
			log.debug(f"Reading ELPH data serially. NQ: {p.N_Q}, NK: {p.N_k}, Expected Bands: {actual_bands}, Expected Modes: {actual_nmodes}")

		for fragment_ind in range(p.N_Q):
			iQ = fragment_ind

			fname = f"ndb.elph_gkkp_expanded_fragment_{fragment_ind + 1}"
			file_path = os.path.join(elph_save_dir, fname)

			if not os.path.isfile(file_path):
				log.warning(f"ELPH file not found: {file_path}. Skipping Q-index {iQ}.")
				continue

			ds = Dataset(file_path, 'r')
			elph_data_frag_raw = ds.variables[f"ELPH_GKKP_Q{fragment_ind + 1}"][:]
			ph_freqs_frag_raw = ds.variables[f"PH_FREQS{fragment_ind + 1}"][:]
			ds.close()

			num_k_in_file = elph_data_frag_raw.shape[0]
			if num_k_in_file != p.N_k:
				log.warning(f"ELPH Fragment {fragment_ind+1} has {num_k_in_file} k-points, expected {p.N_k}. Will use data up to min(p.N_k, {num_k_in_file}).")

			num_k_to_process = min(num_k_in_file, p.N_k)

			elph_complex = elph_data_frag_raw[:num_k_to_process, :actual_bands, :actual_bands, :actual_nmodes, 0] + \
						   1j * elph_data_frag_raw[:num_k_to_process, :actual_bands, :actual_bands, :actual_nmodes, 1]

			g_elph[iQ, :num_k_to_process, :, :, :] = elph_complex.astype(np.complex64)
			ph_freq[iQ, :actual_nmodes] = ph_freqs_frag_raw[:actual_nmodes].astype(np.float32)

	if rank == ROOT:
		log.info(f"g_elph shape: {g_elph.shape}")
		log.info(f"ph_freq shape: {ph_freq.shape}")
	return g_elph, ph_freq

def normalize_elph(g_elph, ph_freq):
	"""
	Normalize the electron-phonon matrix g_elph by sqrt(2 * omega_q_mode).
	Uses constants meV and eps_acustic from common.constants.
	"""

	ph_freq_abs_sqrt = np.sqrt(np.abs(ph_freq))

	if rank == ROOT:

		if ph_freq_abs_sqrt.shape[0] > 0 and ph_freq_abs_sqrt.shape[1] > 0:
			log.info(f'Phonon frequencies @ Gamma (meV) before normalization (abs_sqrt): {ph_freq_abs_sqrt[0,:] / meV}')
		else:
			log.info('Phonon frequency array is empty or too small to show Gamma point.')


	num_modes_in_data = g_elph.shape[-1]

	for q_idx in range(p.N_Q):
		for l_idx in range(num_modes_in_data):
			current_ph_freq_val = ph_freq_abs_sqrt[q_idx, l_idx]
			if np.abs(current_ph_freq_val) > eps_acustic:
				g_elph[q_idx, :, :, :, l_idx] /= np.sqrt(2.0 * current_ph_freq_val)
			else:
				if rank == ROOT:
					log.warning(f"Small phonon frequency at Q={q_idx}, mode={l_idx}: {current_ph_freq_val:.4e} < {eps_acustic:.4e}. Setting g_elph for this mode to 0.")
				g_elph[q_idx, :, :, :, l_idx] = 0.0

	if rank == ROOT:
		log.info(f'Electron-phonon matrix g_elph shape after normalization: {g_elph.shape}')

	return g_elph, ph_freq_abs_sqrt

def save_Geph(G, iQ1, iQ2, excph_path):
	"""Save computed results to files."""
	for Q_ind in range(iQ1, iQ2):
		G[Q_ind-iQ1].tofile(excph_path+f'/excph_Q{Q_ind + 1}.dat')
