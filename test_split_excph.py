#!/usr/bin/env python3
"""
Test script for the split_excph functionality.
This script tests the new split_excph module to ensure it works correctly.

Usage:
    # Serial test
    python test_split_excph.py -yml_inp config.yml

    # MPI test
    mpiexec -n 2 python test_split_excph.py -yml_inp config.yml
"""

import sys
import os
import numpy as np
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from logmod import log
from common.param import p
from mpi_module import mpi
from split_excph import run_split_excph, read_excph_input_data

def create_test_data():
    """Create minimal test data for split_excph testing."""
    if mpi.rank == mpi.root:
        log.info("Creating test data...")

        # Create output directory
        os.makedirs(p.output_data_dir, exist_ok=True)

        # Create minimal test arrays
        # Small dimensions for testing
        test_N_Q = min(4, p.N_Q)
        test_N_k = min(4, p.N_k)
        test_alpha = min(2, p.alpha)
        test_beta = min(2, p.beta)
        test_nmodes = min(3, p.nmodes)
        test_N_v = min(1, p.N_v)
        test_N_c = min(1, p.N_c)

        # Update parameters for test
        p.N_Q = test_N_Q
        p.N_k = test_N_k
        p.alpha = test_alpha
        p.beta = test_beta
        p.nmodes = test_nmodes
        p.N_v = test_N_v
        p.N_c = test_N_c
        p.N_q = test_N_Q

        log.info(f"Test parameters: N_Q={p.N_Q}, N_k={p.N_k}, alpha={p.alpha}, beta={p.beta}")

        # Create test g_elph array
        g_elph_shape = (p.N_Q, p.N_k, p.N_v + p.N_c, p.N_v + p.N_c, p.nmodes)
        g_elph = np.random.random(g_elph_shape).astype(np.complex64)
        g_elph += 1j * np.random.random(g_elph_shape).astype(np.complex64)

        # Create test A_exc array
        A_exc_shape = (p.N_Q, p.beta, p.N_k, p.N_v, p.N_c)
        A_exc = np.random.random(A_exc_shape).astype(np.complex64)
        A_exc += 1j * np.random.random(A_exc_shape).astype(np.complex64)

        # Save test data
        g_elph_file = os.path.join(p.output_data_dir, 'g_elph.dat')
        A_exc_file = os.path.join(p.output_data_dir, 'A_exc.dat')

        g_elph.tofile(g_elph_file)
        A_exc.tofile(A_exc_file)

        log.info(f"Created test g_elph.dat with shape: {g_elph.shape}")
        log.info(f"Created test A_exc.dat with shape: {A_exc.shape}")

        return True

    return False

def verify_fragments():
    """Verify that the fragment files were created correctly."""
    if mpi.rank == mpi.root:
        log.info("Verifying fragment files...")

        excph_dir = os.path.join(p.output_data_dir, 'excph_dir')

        if not os.path.exists(excph_dir):
            log.error(f"Fragment directory not found: {excph_dir}")
            return False

        # Check that all fragment files exist
        missing_files = []
        fragment_sizes = []

        for Q_ind in range(p.N_Q):
            filename = os.path.join(excph_dir, f'excph_Q{Q_ind + 1}.dat')

            if not os.path.exists(filename):
                missing_files.append(filename)
            else:
                # Check file size
                file_size = os.path.getsize(filename)
                fragment_sizes.append(file_size)

                # Try to read the fragment
                try:
                    data = np.fromfile(filename, dtype='complex64')
                    expected_size = p.N_q * p.alpha * p.beta * p.nmodes

                    if data.size != expected_size:
                        log.warning(f"Fragment {Q_ind+1} has size {data.size}, expected {expected_size}")
                    else:
                        log.debug(f"Fragment {Q_ind+1} verified: {data.size} elements")

                except Exception as e:
                    log.error(f"Error reading fragment {Q_ind+1}: {e}")
                    missing_files.append(filename)

        if missing_files:
            log.error(f"Missing fragment files: {missing_files}")
            return False

        # Check file sizes are consistent
        if len(set(fragment_sizes)) > 1:
            log.warning(f"Fragment files have different sizes: {set(fragment_sizes)}")

        log.info(f"✓ All {p.N_Q} fragment files created successfully")
        log.info(f"✓ Fragment size: {fragment_sizes[0] if fragment_sizes else 0} bytes")

        return True

    return False

def test_data_reading():
    """Test the data reading functionality."""
    if mpi.rank == mpi.root:
        log.info("Testing data reading...")

    # Test reading the data
    g_elph, A_exc = read_excph_input_data()

    if g_elph is None or A_exc is None:
        if mpi.rank == mpi.root:
            log.error("✗ Data reading failed")
        return False

    if mpi.rank == mpi.root:
        log.info(f"✓ Successfully read g_elph with shape: {g_elph.shape}")
        log.info(f"✓ Successfully read A_exc with shape: {A_exc.shape}")

    return True

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test split_excph functionality')
    parser.add_argument('-yml_inp', nargs=1, required=True, help='Configuration YAML file')
    args = parser.parse_args()

    # Initialize parameters
    yml_input = args.yml_inp[0]
    if mpi.rank == mpi.root:
        log.info("="*60)
        log.info("SPLIT_EXCPH TEST")
        log.info("="*60)
        log.info(f"Using configuration: {yml_input}")

        if not os.path.exists(yml_input):
            log.error(f"Configuration file not found: {yml_input}")
            return 1

    # Read configuration
    p.read_input_parameters(yml_input)

    if mpi.rank == mpi.root:
        log.info(f"MPI processes: {mpi.size}")
        log.info(f"Original parameters: N_Q={p.N_Q}, N_k={p.N_k}")

    try:
        # Step 1: Create test data
        if not create_test_data():
            if mpi.rank == mpi.root:
                log.error("Failed to create test data")
            return 1

        # Broadcast updated parameters to all processes
        if mpi.size > 1:
            p_dict = {
                'N_Q': p.N_Q, 'N_k': p.N_k, 'alpha': p.alpha, 'beta': p.beta,
                'nmodes': p.nmodes, 'N_v': p.N_v, 'N_c': p.N_c, 'N_q': p.N_q
            }
            p_dict = mpi.comm.bcast(p_dict, root=mpi.root)

            if mpi.rank != mpi.root:
                p.N_Q = p_dict['N_Q']
                p.N_k = p_dict['N_k']
                p.alpha = p_dict['alpha']
                p.beta = p_dict['beta']
                p.nmodes = p_dict['nmodes']
                p.N_v = p_dict['N_v']
                p.N_c = p_dict['N_c']
                p.N_q = p_dict['N_q']

        # Step 2: Test data reading
        if not test_data_reading():
            if mpi.rank == mpi.root:
                log.error("Data reading test failed")
            return 1

        # Step 3: Run split_excph
        if mpi.rank == mpi.root:
            log.info("Running split_excph calculation...")

        success = run_split_excph()

        if not success:
            if mpi.rank == mpi.root:
                log.error("✗ split_excph calculation failed")
            return 1

        # Step 4: Verify results
        if not verify_fragments():
            if mpi.rank == mpi.root:
                log.error("Fragment verification failed")
            return 1

        if mpi.rank == mpi.root:
            log.info("="*60)
            log.info("✓ ALL TESTS PASSED")
            log.info("="*60)
            log.info("Test Results:")
            log.info(f"  - Created test data for {p.N_Q} Q-points")
            log.info(f"  - Successfully computed {p.N_Q} fragments")
            log.info(f"  - All fragments verified")
            log.info(f"  - MPI processes: {mpi.size}")
            log.info("="*60)

        return 0

    except Exception as e:
        if mpi.rank == mpi.root:
            log.error(f"Test failed with error: {str(e)}")
            import traceback
            log.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
