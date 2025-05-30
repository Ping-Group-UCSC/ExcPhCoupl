from mpi4py import MPI
import numpy as np

#  ROOT

MPI_ROOT = 0

#  MPI class

class MPI_obj:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.root = MPI_ROOT
    def finalize_procedure(self):
        MPI.Finalize()
    def split_range(self, n):
        k, m = divmod(n, self.size)
        res = []
        for i in range(self.size):
            res.append([np.int32(i*k+min(i, m)), np.int32((i+1)*k+min(i+1, m))])
        return res

# MPI obj.

mpi = MPI_obj()