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

# Importing specific functions and parameters from user-defined modules
from common.func import *


# ──────────────────────────────────────────────────────────────────────────────
# 1) Hard-code all “SAVE” paths here: I am testing this with git push
# ──────────────────────────────────────────────────────────────────────────────
BSE_SAVE_DIR   = "/Users/keyneshdongol/Downloads/excph-devel(1)/yambo-qe-nk662-exp-lat/exciton/SAVE"
ELPH_SAVE_DIR  = "/Users/keyneshdongol/Downloads/excph-devel(1)/yambo-qe-nk662-exp-lat/QPT6/dvscf/bn.save/SAVE"
DATA_DIR       = "/Users/keyneshdongol/Downloads/excph-devel(1)/yambo-qe-nk662-exp-lat/QPT6/dvscf/bn.save"
#    ^ this is where qpoints_yambo lives


# MPI setup
comm = mpi.comm
rank = mpi.rank #Rank 0 as “master” for logging and directory creation
size = mpi.size #is the total number of MPI processes.
ROOT = MPI_ROOT

def read_qpoints_yambo():
    """ Rank 0 reads qpoints_yambo, then bcasts the N_Qx3 array to all ranks. """
    qfile = os.path.join(DATA_DIR, "qpoints_yambo")
    if rank == ROOT:
        # Load the file
        with open(qfile, "r") as f:
            lines = f.readlines()
        Q_yambo = np.zeros((p.N_Q, 3), dtype=float)
        for i in range(p.N_Q):
            parts = lines[i+1].split()
            Q_yambo[i,0] = float(parts[0])
            Q_yambo[i,1] = float(parts[1])
            Q_yambo[i,2] = float(parts[2])
    else:
        Q_yambo = None

    # Broadcast so every rank ends up with the full array
    Q_yambo = comm.bcast(Q_yambo, root=ROOT)
    return Q_yambo

def read_bse_wavefunction(Q_yambo):
    """
    Each rank reads only its subset of BSE files in BSE_SAVE_DIR,
    then gather them on ROOT into full A_exc, exc_freq.
    """
    # Determine which Q‐indices this rank handles
    k, m = divmod(p.N_Q, size)
    start = rank * k + min(rank, m)
    end   = start + k + (1 if rank < m else 0)

    # On rank 0, allocate the global arrays; others can start as None
    if rank == ROOT:
        A_exc    = np.zeros((p.N_Q, p.beta, p.N_k, p.N_v, p.N_c), dtype=np.complex128)
        exc_freq = np.zeros((p.N_Q, p.beta), dtype=float)
    else:
        A_exc    = None
        exc_freq = None

    # Each rank reads files for fragment indices [start, end)
    local_list_A    = []  # will hold tuples (iQ, A_slice)
    local_list_freq = []  # will hold tuples (iQ, freq_slice)

    for frag in range(start, end):
        fname = f"ndb.BS_diago_Q{frag+1}"
        path  = os.path.join(BSE_SAVE_DIR, fname)
        # (Optional) you can check existence:
        if rank == ROOT:
            print(f"[rank {rank}] Opening BSE file: {path}")
        ds = Dataset(path, "r")
        BS_EIGENSTATES = ds.variables["BS_EIGENSTATES"][:]  # shape (β, Nk*(Nv+Nc), 2)
        BS_Energies    = ds.variables["BS_Energies"][:]     # shape (β, ...)
        ds.close()

        # Reconstruct complex eigenstates, then reshape:
        #    BS_EIGENSTATES[..., 0] = real part, [...,1] = imag part
        complex_states = BS_EIGENSTATES[:p.beta, :, 0] + 1j*BS_EIGENSTATES[:p.beta, :, 1]
        # Now reshape into (beta, N_k, N_v, N_c)
        A_slice = complex_states.reshape((p.beta, p.N_k, p.N_v, p.N_c))
        freq_slice = BS_Energies[:p.beta, 0]

        local_list_A.append((frag, A_slice))
        local_list_freq.append((frag, freq_slice))

    # Gather all local_list_A and local_list_freq on rank 0
    gathered_A    = comm.gather(local_list_A,    root=ROOT)
    gathered_freq = comm.gather(local_list_freq, root=ROOT)

    if rank == ROOT:
        # Populate the global arrays
        for rdata in gathered_A:
            for (iQ, A_slice) in rdata:
                A_exc[iQ] = A_slice
        for rdata in gathered_freq:
            for (iQ, f_slice) in rdata:
                exc_freq[iQ] = f_slice

    # Broadcast the assembled A_exc and exc_freq back to all ranks
    A_exc    = comm.bcast(A_exc,    root=ROOT)
    exc_freq = comm.bcast(exc_freq, root=ROOT)
    return A_exc, exc_freq

def read_elph_data(Q_yambo):
    """
    Each rank reads its subset of el-ph fragments in ELPH_SAVE_DIR,
    then gather on ROOT into full g_elph and ph_freq.
    We now infer the true band count and number of modes from fragment 1.
    """
    if rank == ROOT:
        sample_fname = os.path.join(ELPH_SAVE_DIR, "ndb.elph_gkkp_expanded_fragment_1")
        if not os.path.isfile(sample_fname):
            raise FileNotFoundError(f"Missing sample el-ph file: {sample_fname}")
        ds_sample = Dataset(sample_fname, "r")
        sample_elph = ds_sample.variables["ELPH_GKKP_Q1"][:]
        ds_sample.close()

        actual_bands  = sample_elph.shape[1]
        actual_nmodes = sample_elph.shape[3]

        g_elph  = np.zeros((p.N_Q, p.N_k, actual_bands, actual_bands, actual_nmodes),
                           dtype=np.complex128)
        ph_freq = np.zeros((p.N_Q, actual_nmodes), dtype=float)
    else:
        actual_bands  = None
        actual_nmodes = None
        g_elph  = None
        ph_freq = None

    # Broadcast actual_bands and actual_nmodes so every rank knows them
    actual_bands  = comm.bcast(actual_bands,  root=ROOT)
    actual_nmodes = comm.bcast(actual_nmodes, root=ROOT)

    # 2) Each rank decides which fragments [start, end) to read
    k, m = divmod(p.N_Q, size)
    start = rank * k + min(rank, m)
    end   = start + k + (1 if rank < m else 0)

    local_g_list  = []
    local_ph_list = []

    for frag in range(start, end):
        fname = f"ndb.elph_gkkp_expanded_fragment_{frag+1}"
        path  = os.path.join(ELPH_SAVE_DIR, fname)
        if rank == ROOT and not os.path.isfile(path):
            raise FileNotFoundError(f"Expected el-ph file not found: {path}")
        ds = Dataset(path, "r")

        # Read the 5D array: (p.N_k, actual_bands, actual_bands, actual_nmodes, 2)
        ELPH_frag     = ds.variables[f"ELPH_GKKP_Q{frag+1}"][:]   # shape (p.N_k, bands, bands, nmodes, 2)
        PH_FREQS_frag = ds.variables[f"PH_FREQS{frag+1}"][:]      # shape (actual_nmodes,)
        ds.close()

        arr_complex = ELPH_frag[..., 0] + 1j * ELPH_frag[..., 1]

        for kfrag in range(p.N_k):
            ik = kfrag
            g_slice  = arr_complex[kfrag]
            ph_slice = PH_FREQS_frag[:actual_nmodes]
            local_g_list.append((frag, ik, g_slice))
            local_ph_list.append((frag, ph_slice))

    gathered_g  = comm.gather(local_g_list,  root=ROOT)
    gathered_ph = comm.gather(local_ph_list, root=ROOT)

    if rank == ROOT:
        for rdata in gathered_g:
            for (iQ, ik, g_slice) in rdata:
                g_elph[iQ, ik] = g_slice
        for rdata in gathered_ph:
            for (iQ, ph_slice) in rdata:
                ph_freq[iQ] = ph_slice

    # 5) Broadcast back to all ranks
    g_elph  = comm.bcast(g_elph,  root=ROOT)
    ph_freq = comm.bcast(ph_freq, root=ROOT)
    return g_elph, ph_freq


def normalize_elph(g_elph, ph_freq):
    """Divide each el-ph matrix by sqrt(2 * ω_q,mode) (with an acoustic cutoff)."""
    eps_acoustic = 1e-3
    meV = 1.0

    # Convert ph_freq to real positive frequencies
    ph_freq = np.sqrt(np.abs(ph_freq))
    if rank == ROOT:
        print("Phonon freq @ Γ (meV) = ", ph_freq[0]/meV)

    # Only normalize those modes > eps_acoustic
    for q_i, l_i in np.ndindex(p.N_Q, p.nmodes):
        if np.abs(ph_freq[q_i, l_i]) > eps_acoustic:
            g_elph[q_i, :, :, :, l_i] /= np.sqrt(2.0 * ph_freq[q_i, l_i])
        else:
            if rank == ROOT:
                print(f"Warning: small phonon freq {ph_freq[q_i,l_i]:.4e} < {eps_acoustic}")
            g_elph[q_i, :, :, :, l_i] = 0.0

    if rank == ROOT:
        print("Electron-phonon matrix shape:", g_elph.shape)
    return g_elph, ph_freq

def save_excph(G):
    """
    Each rank writes out its own slice of G (shape: N_Q × …) as binary .dat files.
    We assume G has shape (N_Q, …), so “split_range” decides which Q‐indices to write.
    """
    k, m = divmod(p.N_Q, size)
    start = rank * k + min(rank, m)
    end   = start + k + (1 if rank < m else 0)

    out_dir = os.path.join(DATA_DIR, "excph_out")
    if rank == ROOT and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    comm.Barrier()  # wait until excph_out exists

    for q_i in range(start, end):
        out_fname = os.path.join(out_dir, f"excph_Q{q_i+1}.dat")
        # Flatten G[q_i] to 1D binary; adjust as needed:
        G[q_i].tofile(out_fname)
        print(f"[rank {rank}] Wrote {out_fname}")

if __name__ == "__main__":
    # 1) Read q-points (rank 0 → broadcast)
    Q_yambo = read_qpoints_yambo()

    # 2) Read BSE wavefunctions in parallel, assemble on ROOT
    A_exc, exc_freq = read_bse_wavefunction(Q_yambo)

    # 3) Read el-ph data in parallel, assemble on ROOT
    g_elph, ph_freq = read_elph_data(Q_yambo)

    # 4) Normalize the electron-phonon coupling
    g_elph, ph_freq = normalize_elph(g_elph, ph_freq)

    # 5) (Example) Suppose G = g_elph (or some function of A_exc & g_elph).
    #    Here we just save g_elph as a stand-in for “exciton-phonon” results:
    save_excph(g_elph)

    # 6) Finalize MPI (mpi4py does this automatically on exit, but just in case):
    mpi.finalize_procedure()
