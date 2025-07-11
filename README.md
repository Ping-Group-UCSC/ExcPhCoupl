# Exciton-Phonon Coupling Package

This post processing package calculates exciton-phonon coupling properties from Yambo and Quantum Espresso.

## Current Status

The core data processing workflow is implemented and unit-tested. This includes:
1.  **Step 1:** Reading raw Yambo/QE NetCDF data.
2.  **Step 2:** Computing exciton-phonon coupling matrices in parallel (`split` and `collect`).
3.  **Step 3:** Phonon Assisted Radiative lifetimes and rates -- In development

## Prerequisites

-   Python 3.xx
-   OpenMPI
-   Required Python packages as listed in `requirements.txt`.

## Quick Start

### 1. Installation

First, install the necessary dependencies from the project root directory:
```bash
pip install -r requirements.txt
```
Then, install the package itself:
```bash
pip install .
```

### 2. Configuration

Edit the `config.yml` file to point to your data directories and set the physical parameters for your system (e.g., `Qmesh`, `N_v`, `N_c`).

### 3. Generate `qpoints_yambo`

Before running the main workflow, you must generate the Q-points file. Run the following utility, ensuring the path inside the script points to your `r_gkkp_gkkp_db` file's directory:
```bash
python utils/get_qlist_yambo.py
```

### 4. Run the Workflow

Execute the main workflow using `mpirun`. This will generate the multiple `.dat` files in the directory specified in your config.
```bash
mpirun -n 4 python src/run.py -yml_inp config.yml
```

<!-- ### 5. Run Tests

To verify the integrity of the code's core logic, run the unit tests using `pytest`:
```bash
# First, generate the small, self-contained test data
python tests/generate_test_data.py

# Then, run the tests
pytest tests/
``` -->

## Testing Strategy

The tests can be found in the `tests/` directory and verify the following:

1.  **K-Point Conversion (`test_kpoint_conversion_roundtrip`)**:
    *   **What:** Ensures that the mapping between k-point vectors and their array indices is mathematically self-consistent.
    *   **Why:** Guarantees the fundamental coordinate transformations are correct.

2.  **Output Shape Verification (`test_compute_excph_contributions_output_shape`)**:
    *   **What:** Confirms that the main computational function produces NumPy arrays with the correct dimensions.
    *   **Why:** Prevents bugs related to array reshaping or incorrect loop boundaries.

3.  **Numerical Stability (`test_compute_excph_contributions_no_nan_or_inf`)**:
    *   **What:** Checks that the calculations do not result in "Not a Number" (`NaN`) or infinite (`inf`) values.
    *   **Why:** Ensures the calculations are numerically stable and free from errors like division by zero.

4.  **Known Value Regression Test (`test_compute_excph_contributions_known_value`)**:
    *   **What:** It runs the main computational function with a simple, known input and verifies that the output matches a pre-calculated, correct answer.
    *   **Why:** Protects the scientific integrity of the code by immediately detecting any accidental changes to the underlying mathematical formulas.

This project is set up with GitHub Actions to automatically run these tests upon every push and pull request to the `main` and `develop` branches.

## Development Highlights

Recent updates have significantly improved the robustness and professionalism of this package:

-   **MPI Parallelization**: The core `split` and `collect` logic has been fully parallelized with MPI, allowing for efficient computation on multi-core systems.
-   **MPI Hang/Deadlock Resolution**: Solved critical bugs related to MPI communication, ensuring that parallel runs complete and terminate cleanly without hanging.
-   **Continuous Integration (CI)**:
    1.  **Serial Unit Tests**: Verifies the core logic and numerical stability.
    2.  **Parallel Integration Test**: Confirms the full application runs without crashing or deadlocking in a multi-process MPI environment.
-   **Code Refactoring**: Improved the code's modularity by refactoring key functions to accept data as arguments rather than reading files directly, making the code cleaner and easier to test.
