# mpi_module.py

import numpy as np

# Define the root process ID, which is conventionally 0.
MPI_ROOT = 0

# --- Graceful MPI Import ---
# This block attempts to import the real mpi4py library. If it fails
# (e.g., in a testing environment where MPI is not fully configured or needed),
# it creates a 'mock' or 'fake' MPI object that allows the code to run in
# a single-process, serial mode without crashing.

try:
    # Attempt to import the actual mpi4py library.
    from mpi4py import MPI
    # Set a flag indicating that we are in a real MPI environment.
    is_mpi_available = True
except ImportError:
    # If the import fails, set the flag to false.
    is_mpi_available = False


class MPI_obj:
    """
    A class that encapsulates MPI functionalities.
    It will use the real mpi4py functionalities if available, otherwise it
    will provide default values for a single-process (serial) execution.
    """
    def __init__(self):
        if is_mpi_available:
            # If mpi4py was imported successfully, use the real MPI communicator.
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            # If mpi4py is not available, create a mock communicator.
            # A 'None' communicator signifies a non-MPI environment.
            self.comm = None
            # In a serial run, the process rank is always 0 (the root).
            self.rank = 0
            # The total number of processes is 1.
            self.size = 1

        # The root process is always 0, in both serial and parallel modes.
        self.root = MPI_ROOT

    def finalize_procedure(self):
        """
        Finalizes the MPI environment if it was initialized.
        """
        if is_mpi_available and self.comm is not None:
            MPI.Finalize()

    def split_range(self, n):
        """
        Splits a range of size 'n' among all available processes.
        This function works correctly for both serial (size=1) and parallel cases.
        """
        # Integer division to determine the base size of each chunk.
        k, m = divmod(n, self.size)
        res = []
        for i in range(self.size):
            # Each process 'i' gets a chunk of size 'k'. The first 'm' processes
            # get one extra element to distribute the remainder.
            start_index = np.int32(i * k + min(i, m))
            end_index = np.int32((i + 1) * k + min(i + 1, m))
            res.append([start_index, end_index])
        return res

# --- Global MPI Object ---
# Create a single, global instance of the MPI_obj class that can be imported
# and used by other modules throughout the application.
# e.g., `from mpi_module import mpi`
mpi = MPI_obj()
