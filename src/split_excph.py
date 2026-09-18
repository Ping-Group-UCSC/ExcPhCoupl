#!/usr/bin/env python3
from __future__ import print_function, division
import numpy as np

import os
import time
from logmod import log
from common.param import p
from common.func import *
from mpi_module import mpi, MPI_ROOT



def _build_kvec_table(N_Q, qmesh):
    """
    Vectorized ik2k for all indices 0..N_Q-1.
    Returns all_kvecs of shape (N_Q, 3).
    """
    iks = np.arange(N_Q, dtype=np.int64)
    kx = iks // (qmesh[1] * qmesh[2])
    ky = (iks // qmesh[2]) % qmesh[1]
    kz = iks % qmesh[2]
    return np.stack([kx / qmesh[0], ky / qmesh[1], kz / qmesh[2]], axis=1)  # (N_Q, 3)


def compute_excph_contributions(g_elph, A_exc, Q_ind):
    """
    Compute exciton-phonon coupling contributions for a single Q point.
    The inner k-point loop is fully vectorized: index maps are precomputed
    and each q iteration uses a single batched einsum over all N_k k-points.
    """
    N_Q = p.N_Q

    # --- Precompute k-vectors and index maps ---
    all_kvecs = _build_kvec_table(N_Q, p.qmesh)   # (N_Q, 3)
    Q_vec = all_kvecs[Q_ind]                        # (3,)

    # Qq_inds[q]   = index of Q + q  (shape N_q)
    Qq_inds = k2ik(Q_vec[None, :] + all_kvecs)     # (N_q,)
    # kQ_inds[k]   = index of k - Q  (shape N_k)
    kQ_inds = k2ik(all_kvecs - Q_vec[None, :])     # (N_k,)
    # kq_table[k,q] = index of k + q (shape N_k x N_q)
    kq_vecs = (all_kvecs[:, None, :] + all_kvecs[None, :, :]).reshape(-1, 3)  # (N_k*N_q, 3)
    kq_table = k2ik(kq_vecs).reshape(N_Q, N_Q)    # (N_k, N_q)

    # Pre-fetch A_exc slice that is constant for all q: shape (alpha, N_k, N_v, N_c)
    A_Q = A_exc[Q_ind, :p.alpha, :, :, :]

    temp_cc = np.zeros((p.N_q, p.alpha, p.beta, p.nmodes), dtype='complex64')
    temp_vv = np.zeros((p.N_q, p.alpha, p.beta, p.nmodes), dtype='complex64')

    log_interval = max(1, p.N_q // 10)

    for q_ind in range(p.N_q):
        if q_ind % log_interval == 0:
            log.debug(f"Rank {mpi.rank}, Q_ind {Q_ind}: q-loop {q_ind}/{p.N_q} "
                      f"({100 * q_ind // p.N_q}%)")

        Qq = int(Qq_inds[q_ind])
        if Qq >= N_Q:
            continue

        kq_k = kq_table[:, q_ind]   # (N_k,) — kq index for each k at this q

        # ── Conduction band contribution ──────────────────────────────────────
        # Per-k original: einsum('mij,nik,jkl->nml',
        #   A_exc[Qq,:beta,kq,:,:].conj(),          # m,i,j
        #   A_exc[Q, :alpha,k,:,:],                 # n,i,a  (a = c-band)
        #   g_elph[q,kq,N_v:,N_v:,:].conj())        # j,a,l
        # Vectorized over K (k-point): 'mKij,nKia,Kjal->nml'
        A1 = A_exc[Qq, :p.beta, kq_k, :, :].conj()                            # (beta, N_k, N_v, N_c)
        G  = g_elph[q_ind, kq_k,
                    p.N_v:p.N_v + p.N_c, p.N_v:p.N_v + p.N_c, :].conj()     # (N_k, N_c, N_c, nmodes)
        temp_cc[q_ind] += np.einsum('mKij,nKia,Kjal->nml', A1, A_Q, G,
                                    optimize=True)

        # ── Valence band contribution ─────────────────────────────────────────
        # Per-k original: -einsum('mij,nkj,kil->nml',
        #   A_exc[Qq,:beta,k,:,:].conj(),            # m,i,j  (raw k, not kq!)
        #   A_exc[Q, :alpha,k,:,:],                  # n,a,j  (a = v-band)
        #   g_elph[q,kQ,0:N_v,0:N_v,:].conj())       # a,i,l
        # Vectorized over K: 'mKij,nKaj,Kail->nml'
        A1 = A_exc[Qq, :p.beta, :, :, :].conj()                               # (beta, N_k, N_v, N_c)
        G  = g_elph[q_ind, kQ_inds, 0:p.N_v, 0:p.N_v, :].conj()             # (N_k, N_v, N_v, nmodes)
        temp_vv[q_ind] += -np.einsum('mKij,nKaj,Kail->nml', A1, A_Q, G,
                                     optimize=True)

    return temp_cc, temp_vv

def save_excph_fragment(Q_ind, temp_cc, temp_vv):
    """
    Save the computed contributions for a single Q point.
    Creates excph_dir if it doesn't exist and saves the fragment.
    """
    excph_dir = os.path.join(p.output_data_dir, 'excph_dir')

    # Create directory if it doesn't exist (only root process)
    if mpi.rank == MPI_ROOT:
        os.makedirs(excph_dir, exist_ok=True)

    # Wait for directory creation
    if mpi.size > 1:
        mpi.comm.Barrier()

    Gamma_Q = temp_cc + temp_vv

    # Save to file
    filename = os.path.join(excph_dir, f'excph_Q{Q_ind + 1}.dat')
    Gamma_Q.tofile(filename)

    log.debug(f"Rank {mpi.rank}: Saved Q-point {Q_ind + 1} to {filename}")

def split_excph_parallel(g_elph, A_exc):
    """
    Parallel computation of exciton-phonon coupling fragments.
    Each MPI process handles a subset of Q-points.
    """
    if mpi.rank == MPI_ROOT:
        log.info("Starting parallel exciton-phonon coupling computation...")
        log.info(f"Processing {p.N_Q} Q-points across {mpi.size} MPI processes")

    # Distribute Q-points among MPI processes
    iQ_lim = mpi.split_range(p.N_Q)
    [iQ1, iQ2] = iQ_lim[mpi.rank]

    log.info(f"Rank {mpi.rank}: processing Q-points {iQ1+1} to {iQ2}")

    # Process assigned Q-points
    for Q_ind in range(iQ1, iQ2):
        log.debug(f"Rank {mpi.rank}: Processing Q-point {Q_ind + 1}/{p.N_Q}")

        # Compute contributions for this Q point
        temp_cc, temp_vv = compute_excph_contributions(g_elph, A_exc, Q_ind)

        # Save the results
        save_excph_fragment(Q_ind, temp_cc, temp_vv)

    # Synchronize all processes
    mpi.comm.Barrier()

    if mpi.rank == MPI_ROOT:
        log.info("Parallel exciton-phonon coupling computation completed!")
        log.info(f"Generated {p.N_Q} excph fragment files in {os.path.join(p.output_data_dir, 'excph_dir')}")

def split_excph_serial(g_elph, A_exc):
    """
    Serial computation of exciton-phonon coupling fragments.
    Used when running without MPI or for debugging.
    """
    if mpi.rank == MPI_ROOT:
        log.info("Starting serial exciton-phonon coupling computation...")
        log.info(f"Processing {p.N_Q} Q-points")

        # Create output directory
        excph_dir = os.path.join(p.output_data_dir, 'excph_dir')
        os.makedirs(excph_dir, exist_ok=True)

        # Process each Q point
        for Q_ind in range(p.N_Q):
            log.debug(f"Processing Q-point {Q_ind + 1}/{p.N_Q}")

            temp_cc, temp_vv = compute_excph_contributions(g_elph, A_exc, Q_ind)

            save_excph_fragment(Q_ind, temp_cc, temp_vv)

        log.info("Serial exciton-phonon coupling computation completed!")
        log.info(f"Generated {p.N_Q} excph fragment files in {excph_dir}")

def run_split_excph(g_elph, A_exc):
    """
    Main function to run the split exciton-phonon coupling calculation.
    Accepts data arrays as arguments and automatically detects MPI environment
    to run the appropriate parallel or serial version.
    """
    start_time = time.time()

    if mpi.rank == MPI_ROOT:
        log.info("="*80)
        log.info("EXCITON-PHONON COUPLING SPLIT CALCULATION")
        log.info("="*80)
        log.info(f"Expected parameters:")
        log.info(f"  N_Q: {p.N_Q}")
        log.info(f"  N_q: {p.N_q}")
        log.info(f"  N_k: {p.N_k}")
        log.info(f"  alpha: {p.alpha}")
        log.info(f"  beta: {p.beta}")
        log.info(f"  nmodes: {p.nmodes}")
        log.info(f"  N_v: {p.N_v}")
        log.info(f"  N_c: {p.N_c}")
        log.info("="*80)

    # Check if input data is valid
    if g_elph is None or A_exc is None:
        if mpi.rank == MPI_ROOT:
            log.error("Received invalid data arrays. Exiting split_excph.")
        return False

    # Run computation
    if mpi.size > 1:
        split_excph_parallel(g_elph, A_exc)
    else:
        split_excph_serial(g_elph, A_exc)

    # Report timing
    total_time = time.time() - start_time
    if mpi.rank == MPI_ROOT:
        log.info(f"Total execution time: {total_time:.2f} seconds")
        log.info("Next step: Run collection to combine all fragments")

    return True

# For standalone execution
if __name__ == "__main__":
    success = run_split_excph()
    if not success:
        exit(1)
