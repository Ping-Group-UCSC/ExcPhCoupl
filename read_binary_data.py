#!/usr/bin/env python3
"""
Utility script to read and compare binary .dat files.

This script can be used in two modes:
1.  **Inspect Mode:** Provide one file path to read it, reshape it, and see a summary.
2.  **Compare Mode:** Provide two file paths to load both and compare them for
    numerical identity.

Usage:
    # --- Inspect a single file ---
    python read_binary_data.py <path/to/your/file.dat>

    # --- Compare two files ---
    python read_binary_data.py <path/to/file1.dat> <path/to/file2.dat>

Example:
    python read_binary_data.py ./calculation_output/output_files/excph.dat /Users/keyneshdongol/YexcphPP/excph.dat
"""

import numpy as np
import os
import sys
import argparse

# Add src directory to path to import parameters
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from common.param import p

def get_file_info(filename):
    """
    Returns the expected shape and data type for a given binary file based on its name.
    """
    basename = os.path.basename(filename)
    # This dictionary maps filenames to their properties.
    file_map = {
        'excph.dat': {
            'shape': (p.N_Q, p.N_q, p.alpha, p.beta, p.nmodes),
            'dtype': np.complex64,
            'description': 'Exciton-Phonon Coupling Matrix (G)'
        },
        'excph2.dat': {
            'shape': (p.N_Q, p.N_q, p.alpha, p.beta, p.nmodes),
            'dtype': np.float32,
            'description': 'Squared Magnitude of Exciton-Phonon Coupling Matrix (|G|^2)'
        },
        'A_exc.dat': {
            'shape': (p.N_Q, p.beta, p.N_k, p.N_v, p.N_c),
            'dtype': np.complex64,
            'description': 'Exciton Wavefunctions (A_exc)'
        },
        'exc_freq.dat': {
            'shape': (p.N_Q, p.beta),
            'dtype': np.float32,
            'description': 'Exciton Frequencies'
        },
        'exc_freq_deg.dat': {
            'shape': (p.N_Q, p.beta),
            'dtype': np.float32,
            'description': 'Degenerate Exciton Frequencies'
        },
        'g_elph.dat': {
            'shape': (p.N_Q, p.N_k, p.N_v + p.N_c, p.N_v + p.N_c, p.nmodes),
            'dtype': np.complex64,
            'description': 'Electron-Phonon Coupling Matrix (g_elph)'
        },
        'ph_freq.dat': {
            'shape': (p.N_Q, p.nmodes),
            'dtype': np.float32,
            'description': 'Phonon Frequencies'
        },
    }
    return file_map.get(basename)

def load_and_reshape(filepath):
    """
    Loads a single binary file and reshapes it. Returns the reshaped array or None on error.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return None

    file_info = get_file_info(filepath)
    if not file_info:
        print(f"Warning: No shape/dtype information for '{os.path.basename(filepath)}'. Cannot reshape.")
        return np.fromfile(filepath) # Return as flat array

    shape = file_info['shape']
    dtype = file_info['dtype']

    try:
        data_flat = np.fromfile(filepath, dtype=dtype)
        expected_elements = np.prod(shape)

        if data_flat.size != expected_elements:
            print(f"Error: Element count mismatch for '{os.path.basename(filepath)}'.")
            print(f"       File has {data_flat.size} elements, but shape {shape} requires {expected_elements}.")
            return None

        return data_flat.reshape(shape)
    except Exception as e:
        print(f"An error occurred loading {filepath}: {e}")
        return None

def main():
    """
    Main function to parse arguments and perform inspection or comparison.
    """
    parser = argparse.ArgumentParser(description="Inspect or compare binary .dat files.")
    parser.add_argument('files', nargs='+', help="One or two file paths to inspect or compare.")
    parser.add_argument(
        '-c', '--config',
        default='config.yml',
        help="Path to the configuration file to load parameters from (default: config.yml)."
    )
    args = parser.parse_args()

    # Load parameters to get shape information
    if not os.path.exists(args.config):
        print(f"Config file '{args.config}' not found.")
        sys.exit(1)
    p.read_input_parameters(args.config)

    num_files = len(args.files)

    if num_files == 1:
        # --- Inspect Mode ---
        filepath = args.files[0]
        print(f"--- Inspecting File: {os.path.basename(filepath)} ---")
        data = load_and_reshape(filepath)
        if data is not None:
            print(f"Shape: {data.shape}")
            print(f"Data Type: {data.dtype}")
            print("Sample (first element):")
            print(data.flat[0])

    elif num_files == 2:
        # --- Compare Mode ---
        file1_path = args.files[0]
        file2_path = args.files[1]
        print(f"--- Comparing Files ---")
        print(f"File 1: {file1_path}")
        print(f"File 2: {file2_path}")
        print("-----------------------")

        array1 = load_and_reshape(file1_path)
        array2 = load_and_reshape(file2_path)

        if array1 is None or array2 is None:
            print("Cannot perform comparison due to loading errors.")
            sys.exit(1)

        if array1.shape != array2.shape:
            print("Result: SHAPES MISMATCH")
            print(f"  Shape 1: {array1.shape}")
            print(f"  Shape 2: {array2.shape}")
            sys.exit(1)
        else:
            print("Shapes Match:", array1.shape)

        # Perform numerical comparison using numpy.allclose
        # This is better than a direct == comparison for floating point numbers.
        if np.allclose(array1, array2):
            print("Result: NUMERICALLY IDENTICAL")
            print("The files are considered the same within standard tolerance.")
        else:
            print("Result: NUMERICAL DIFFERENCE DETECTED")
            # Find the location of the first difference
            diff_indices = np.where(np.isclose(array1, array2) == False)
            first_diff_index = tuple(idx[0] for idx in diff_indices)
            val1 = array1[first_diff_index]
            val2 = array2[first_diff_index]
            print("First difference found at index:", first_diff_index)
            print(f"  Value in File 1: {val1}")
            print(f"  Value in File 2: {val2}")

    else:
        print("Error: Please provide either one or two file paths.")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
