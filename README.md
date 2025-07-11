# Exciton-Phonon Coupling Package

This package calculates exciton-phonon coupling properties from Yambo and Quantum Espresso simulation data, featuring an MPI-parallelized workflow.

## Current Status

The core data processing workflow is implemented and unit-tested. This includes:
1.  **Step 1:** Reading raw Yambo/QE NetCDF data.
2.  **Step 2:** Computing exciton-phonon coupling matrices in parallel (`split` and `collect`).
3.  **Step 3:** Averaging degenerate exciton energies.

The project includes a full Continuous Integration (CI) setup using unit tests to ensure the mathematical correctness of the core computational logic.

## Prerequisites

-   Python 3.8+
-   An MPI implementation (e.g., OpenMPI, MPICH)
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

Execute the main workflow using `mpirun`. This will run steps 1-3 and generate the final `.dat` files in the directory specified in your config.
```bash
mpirun -n 4 python src/run.py -yml_inp config.yml
```

### 5. Run Tests

To verify the integrity of the code's core logic, run the unit tests using `pytest`:
```bash
# First, generate the small, self-contained test data
python tests/generate_test_data.py

# Then, run the tests
pytest tests/
```

This project is set up with GitHub Actions to automatically run these tests upon every push and pull request to the `main` branch.
