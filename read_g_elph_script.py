# read_g_elph_script.py
import numpy as np
import os


# Path to the file
g_elph_file_path = "./calculation_output/output_files/g_elph.dat"

# Data type
dtype = np.complex64

# Shape parameters
N_Q = 128
N_k = 128
N_v = 2
N_c = 2
actual_bands = N_v + N_c
nmodes = 12

expected_shape = (N_Q, N_k, actual_bands, actual_bands, nmodes)


if not os.path.exists(g_elph_file_path):
    print(f"Error: File not found at {g_elph_file_path}")
else:
    print(f"Attempting to read: {g_elph_file_path}")
    print(f"Expected dtype: {dtype}")
    print(f"Expected shape: {expected_shape}")

    try:
        g_elph_flat = np.fromfile(g_elph_file_path, dtype=dtype)

        g_elph_reshaped = g_elph_flat.reshape(expected_shape)

        print(f"Successfully read and reshaped g_elph.dat.")
        print(f"Shape of the loaded array: {g_elph_reshaped.shape}")

        print("\n--- Example Data ---")
        print("Element at index (0,0,0,0,0):")
        print(g_elph_reshaped[0, 0, 0, 0, 0])

        if N_Q > 0 and N_k > 0 and actual_bands > 0 and nmodes > 0:
            print("\nSlice g_elph[0, 0, 0, :min(2,actual_bands), :min(2,nmodes)]:")
            print(g_elph_reshaped[0, 0, 0, :min(2,actual_bands), :min(2,nmodes)])

        print("\n--- Statistics ---")
        print(f"Mean (real part): {np.mean(g_elph_reshaped.real):.4e}")
        print(f"Mean (imaginary part): {np.mean(g_elph_reshaped.imag):.4e}")
        print(f"Max absolute value: {np.max(np.abs(g_elph_reshaped)):.4e}")
        print(f"Min absolute value: {np.min(np.abs(g_elph_reshaped)):.4e}")

    except Exception as e:
        print(f"An error occurred while reading or reshaping the file: {e}")
        print("Please double-check the dtype and shape parameters against the ones used to create the file.")
