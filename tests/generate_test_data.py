#!/usr/bin/env python3
"""
Generates and saves small, predictable test data arrays.

This script creates a small set of NumPy arrays with known shapes and values,
which are then used as inputs for the unit tests. This ensures that the tests
run with consistent, version-controlled data, making them reliable and
repeatable.

This script only needs to be run once to generate the test data, or if you
decide to change the fundamental dimensions of the test inputs.
"""

import numpy as np
import os
import sys
from netCDF4 import Dataset

# Add src directory to path to import parameters. This allows us to use the
# parameter object to define the dimensions of our test data.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from common.param import p

# --- Configuration for Test Data Generation ---

# Define the directory where the test input files will be saved.
TEST_INPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_inputs')

# Define small, manageable dimensions for our test arrays.
# These dimensions are much smaller than a real production run, making the
# unit tests fast.
TEST_N_Q = 4    # Number of Q-points
TEST_N_K = 4    # Number of k-points (must match N_Q for this test setup)
TEST_N_V = 2    # Number of valence bands
TEST_N_C = 2    # Number of conduction bands
TEST_ALPHA = 2  # Number of initial exciton states
TEST_BETA = 2   # Number of final exciton states
TEST_NMODES = 3 # Number of phonon modes

def generate_and_save_data():
    """
    Creates and saves the test input arrays to the 'test_inputs' directory.
    """
    print(f"Generating test data in: {TEST_INPUT_DIR}")
    os.makedirs(TEST_INPUT_DIR, exist_ok=True)

    # --- Override the default parameters with our smaller test dimensions ---
    # This ensures that any part of the code that imports 'p' during the tests
    # will use these small, fast dimensions.
    p.N_Q = TEST_N_Q
    p.N_k = TEST_N_K
    p.N_q = TEST_N_Q  # In this simplified model, N_q is the same as N_Q
    p.N_v = TEST_N_V
    p.N_c = TEST_N_C
    p.alpha = TEST_ALPHA
    p.beta = TEST_BETA
    p.nmodes = TEST_NMODES

    # --- Create a predictable g_elph array ---
    # The shape is based on the test parameters.
    # We use np.arange to create a predictable sequence of numbers, rather than
    # random data, which makes debugging easier. The values are complex numbers.
    g_elph_shape = (p.N_Q, p.N_k, p.N_v + p.N_c, p.N_v + p.N_c, p.nmodes)
    g_elph_size = np.prod(g_elph_shape)
    g_elph_flat = (np.arange(g_elph_size) * 0.01) + 1j * (np.arange(g_elph_size, 0, -1) * 0.01)
    g_elph_array = g_elph_flat.reshape(g_elph_shape).astype(np.complex64)

    # --- Create a predictable A_exc array ---
    # The shape is also based on the test parameters.
    A_exc_shape = (p.N_Q, p.beta, p.N_k, p.N_v, p.N_c)
    A_exc_size = np.prod(A_exc_shape)
    A_exc_flat = (np.arange(A_exc_size) * 0.01) + 1j * (np.arange(A_exc_size, 0, -1) * 0.01)
    A_exc_array = A_exc_flat.reshape(A_exc_shape).astype(np.complex64)

    # --- Save the arrays to binary files ---
    # These files will be checked into your git repository and used by the unit tests.
    g_elph_path = os.path.join(TEST_INPUT_DIR, 'g_elph.dat')
    A_exc_path = os.path.join(TEST_INPUT_DIR, 'A_exc.dat')

    g_elph_array.tofile(g_elph_path)
    A_exc_array.tofile(A_exc_path)

    # --- Create dummy NetCDF files for BSE data ---
    # The test doesn't need real BSE data, just files with the correct names
    # and variable structure to pass the initial file-opening checks.
    # We loop from 1 to N_Q to create a file for each expected Q-point.
    print(f"Generating {p.N_Q} dummy BSE files...")
    for i in range(1, p.N_Q + 1):
        dummy_bse_path = os.path.join(TEST_INPUT_DIR, f'ndb.BS_diago_Q{i}')
        with Dataset(dummy_bse_path, 'w', format='NETCDF4_CLASSIC') as ds:
            # Define the dimensions based on our test parameters.
            ds.createDimension('complex', 2)
            ds.createDimension('n_states', p.beta)
            ds.createDimension('n_kvc', p.N_k * p.N_v * p.N_c)
            ds.createDimension('n_energies', 2) # Typically Eigen-value and imaginary part

            # Create the variables that the read_io function expects to find.
            bs_eigenstates = ds.createVariable('BS_EIGENSTATES', 'f4', ('n_states', 'n_kvc', 'complex'))
            bs_energies = ds.createVariable('BS_Energies', 'f4', ('n_states', 'n_energies'))

            # Fill the variables with placeholder data (arrays of zeros).
            eigenstates_data = np.zeros((p.beta, p.N_k * p.N_v * p.N_c, 2))
            energies_data = np.zeros((p.beta, 2))
            bs_eigenstates[:] = eigenstates_data
            bs_energies[:] = energies_data

    print("\n--- Test Data Generation Summary ---")
    print(f"g_elph array generated with shape: {g_elph_array.shape}")
    print(f"Saved to: {g_elph_path}")
    print(f"A_exc array generated with shape: {A_exc_array.shape}")
    print(f"Saved to: {A_exc_path}")
    print("\nGeneration complete. You can now run the unit tests.")

if __name__ == "__main__":
    generate_and_save_data()
