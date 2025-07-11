#!/usr/bin/env python3
from __future__ import print_function, division
import numpy as np
import itertools
import os
import time
from logmod import log
from common.param import p
from common.func import *
from mpi_module import mpi, MPI_ROOT



def compute_excph_contributions(g_elph, A_exc, Q_ind):
    """
    Compute exciton-phonon coupling contributions for a single Q point.
    This follows the exact logic from the original split_excph.py script.
    """
    temp_cc = np.zeros((p.N_q, p.alpha, p.beta, p.nmodes), dtype='complex64')
    temp_vv = np.zeros((p.N_q, p.alpha, p.beta, p.nmodes), dtype='complex64')

    # Add a progress counter
    total_iterations = p.N_q * p.N_k
    iteration_count = 0
    log_interval = total_iterations // 10

    for q_ind, k_ind in itertools.product(range(p.N_q), range(p.N_k)):
        iteration_count += 1
        if iteration_count % log_interval == 0:
            progress = (iteration_count / total_iterations) * 100
            log.debug(f"Rank {mpi.rank}, Q_ind {Q_ind}: Loop progress {progress:.1f}% ({iteration_count}/{total_iterations})")

        try:
            # Compute k-point indices
            Qq_ind = k2ik(ik2k(Q_ind) + ik2k(q_ind))
            kq_ind = k2ik(ik2k(k_ind) + ik2k(q_ind))
            kQ_ind = k2ik(ik2k(k_ind) - ik2k(Q_ind))

            # Ensure indices are within bounds
            if Qq_ind >= p.N_Q or kq_ind >= p.N_k or kQ_ind >= p.N_k:
                continue

            # Conduction band contribution
            temp_cc[q_ind, :, :, :] += np.einsum(
                'mij,nik,jkl->nml',
                A_exc[Qq_ind, 0:p.beta, kq_ind, :, :].conj(),
                A_exc[Q_ind, 0:p.alpha, k_ind, :, :],
                g_elph[q_ind, kq_ind, p.N_v:p.N_v + p.N_c, p.N_v:p.N_v + p.N_c, :].conj()
            )

            # Valence band contribution
            temp_vv[q_ind, :, :, :] += -np.einsum(
                'mij,nkj,kil->nml',
                A_exc[Qq_ind, 0:p.beta, k_ind, :, :].conj(),
                A_exc[Q_ind, 0:p.alpha, k_ind, :, :],
                g_elph[q_ind, kQ_ind, 0:p.N_v, 0:p.N_v, :].conj()
            )

        except (IndexError, ValueError) as e:
            log.debug(f"Skipping indices Q={Q_ind}, q={q_ind}, k={k_ind}: {e}")
            continue

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
