#!/usr/bin/env python3
"""
Unit tests for the core logic of the YexcphPP workflow.

This test suite focuses on verifying the correctness of individual functions
(the "units") rather than comparing large, final output files. This makes the
tests more robust, faster, and independent of the specific physical system
(material) being simulated.

Usage:
    # From the project root directory (ExcPhCoupl/):
    pytest tests/
"""

import os
import sys
import numpy as np
import pytest

# Add the source directory to the Python path to allow importing the project's modules.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from common.param import p
from common.func import k2ik, ik2k
from split_excph import compute_excph_contributions

# --- Constants for the Test Setup ---

TEST_DIR = os.path.dirname(__file__)
TEST_INPUT_DIR = os.path.join(TEST_DIR, 'test_inputs')

# Define the small, manageable dimensions used to generate the test data.
# These MUST match the dimensions used in the `generate_test_data.py` script.
TEST_N_Q = 4
TEST_N_K = 4
TEST_N_V = 2
TEST_N_C = 2
TEST_ALPHA = 2
TEST_BETA = 2
TEST_NMODES = 3

# --- Test Fixture ---

@pytest.fixture(scope="module")
def test_data():
    """
    A pytest fixture to load the small, predictable test data from files.

    This fixture runs once per module, loads the test arrays from the
    'test_inputs' directory, and provides them to the test functions. This
    avoids reloading the data for every single test.

    Returns:
        A dictionary containing the loaded 'g_elph' and 'A_exc' NumPy arrays.
    """
    # Override default parameters with our smaller test dimensions.
    # This is crucial for ensuring that functions using 'p' have the correct
    # dimensional information when they are being tested.
    p.N_Q = p.N_q = p.N_k = TEST_N_Q
    p.N_v = TEST_N_V
    p.N_c = TEST_N_C
    p.alpha = TEST_ALPHA
    p.beta = TEST_BETA
    p.nmodes = TEST_NMODES

    # Construct paths to the test data files.
    g_elph_path = os.path.join(TEST_INPUT_DIR, 'g_elph.dat')
    A_exc_path = os.path.join(TEST_INPUT_DIR, 'A_exc.dat')

    # Check that the test data exists. If not, the tests cannot run.
    if not os.path.exists(g_elph_path) or not os.path.exists(A_exc_path):
        pytest.fail(
            "Test data not found. Please run 'tests/generate_test_data.py' first."
        )

    # --- Load g_elph array ---
    g_elph_shape = (p.N_Q, p.N_k, p.N_v + p.N_c, p.N_v + p.N_c, p.nmodes)
    g_elph_array = np.fromfile(g_elph_path, dtype=np.complex64).reshape(g_elph_shape)

    # --- Load A_exc array ---
    A_exc_shape = (p.N_Q, p.beta, p.N_k, p.N_v, p.N_c)
    A_exc_array = np.fromfile(A_exc_path, dtype=np.complex64).reshape(A_exc_shape)

    # Return the loaded data in a dictionary for easy access in tests.
    return {'g_elph': g_elph_array, 'A_exc': A_exc_array}

# --- Unit Tests ---

def test_kpoint_conversion_roundtrip():
    """
    Tests the k-point to index conversion functions (k2ik and ik2k).

    This test verifies that converting an index to a k-point vector and then
    back to an index yields the original index. This confirms the mathematical
    consistency of the coordinate mapping logic.
    """
    print("Testing k-point to index round-trip conversion...")

    # Choose a sample index to test.
    original_index = 10

    # Perform the round-trip conversion.
    k_vector = ik2k(original_index)
    returned_index = k2ik(k_vector)

    # Assert that the returned index is the same as the original.
    assert returned_index == original_index, \
        f"k-point conversion failed: {original_index} -> {k_vector} -> {returned_index}"

    print(f"✅ Round-trip for index {original_index} successful.")

def test_compute_excph_contributions_output_shape(test_data):
    """
    Tests that the main computational function produces outputs of the correct shape.

    Args:
        test_data: The pytest fixture providing the test input arrays.
    """
    print("Testing the output shape of compute_excph_contributions...")

    # Get the test input data from the fixture.
    g_elph = test_data['g_elph']
    A_exc = test_data['A_exc']

    # Define a test Q-point index.
    test_Q_ind = 0

    # Run the function under test.
    temp_cc, temp_vv = compute_excph_contributions(g_elph, A_exc, test_Q_ind)

    # Define the shape we expect the output arrays to have.
    expected_shape = (p.N_q, p.alpha, p.beta, p.nmodes)

    # Assert that the actual shapes match the expected shape.
    assert temp_cc.shape == expected_shape, \
        f"Conduction band output shape is incorrect. Expected {expected_shape}, got {temp_cc.shape}"
    assert temp_vv.shape == expected_shape, \
        f"Valence band output shape is incorrect. Expected {expected_shape}, got {temp_vv.shape}"

    print(f"✅ Output arrays have the correct shape: {expected_shape}")

def test_compute_excph_contributions_no_nan_or_inf(test_data):
    """
    Tests that the computation does not produce any NaN (Not a Number) or
    infinite values, which would indicate a numerical instability.

    Args:
        test_data: The pytest fixture providing the test input arrays.
    """
    print("Testing for NaN or infinite values in the output...")

    g_elph = test_data['g_elph']
    A_exc = test_data['A_exc']
    test_Q_ind = 0

    # Run the function under test.
    temp_cc, temp_vv = compute_excph_contributions(g_elph, A_exc, test_Q_ind)

    # Assert that there are no NaN values in the output arrays.
    assert not np.isnan(temp_cc).any(), "Found NaN values in conduction band output."
    assert not np.isnan(temp_vv).any(), "Found NaN values in valence band output."

    # Assert that there are no infinite values in the output arrays.
    assert np.isfinite(temp_cc).all(), "Found infinite values in conduction band output."
    assert np.isfinite(temp_vv).all(), "Found infinite values in valence band output."

    print("✅ No NaN or infinite values found in outputs.")

def test_compute_excph_contributions_known_value(test_data):
    """
    Tests that a specific element in the output array has a known, correct value.

    This is a powerful regression test. If the underlying math changes, this test
    will fail. We pre-calculate the expected value for one element based on our
    predictable input data.

    Args:
        test_data: The pytest fixture providing the test input arrays.
    """
    print("Testing a specific known value in the output...")

    g_elph = test_data['g_elph']
    A_exc = test_data['A_exc']

    # For this specific test, we will simplify the inputs to get a very predictable output.
    # Set all elements to 1 for simplicity in manual calculation.
    A_exc.fill(1 + 0j)
    g_elph.fill(1 + 0j)

    # Run the function with these simplified inputs.
    # We test for Q_ind = 0.
    temp_cc, temp_vv = compute_excph_contributions(g_elph, A_exc, 0)

    # The calculation involves a sum over N_k, N_v, and N_c.
    # For the conduction part (temp_cc), it's a sum of N_k * N_c terms (since all inputs are 1).
    # For the valence part (temp_vv), it's a sum of N_k * N_v terms.
    # The result of np.einsum is N_c for temp_cc and N_v for temp_vv for each k_ind loop.
    # The final sum is over N_k.
    expected_cc_value = p.N_k * p.N_c
    expected_vv_value = -1 * p.N_k * p.N_v # Note the negative sign for valence.

    # Assert that the first element of the first output array has the expected value.
    # We check the real part because our inputs were real (with 0j imaginary part).
    assert np.isclose(temp_cc[0, 0, 0, 0].real, expected_cc_value), \
        f"Known value test failed for conduction band. Expected ~{expected_cc_value}, got {temp_cc[0,0,0,0].real}"
    assert np.isclose(temp_vv[0, 0, 0, 0].real, expected_vv_value), \
        f"Known value test failed for valence band. Expected ~{expected_vv_value}, got {temp_vv[0,0,0,0].real}"

    print(f"✅ Known value test passed. Expected CC value ~{expected_cc_value}, VV value ~{expected_vv_value}")
