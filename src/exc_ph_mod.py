import numpy as np
import itertools
from common.param import p
from logmod import log
from common.func import *
from mpi_module import mpi, MPI_ROOT

def compute_excph(g_elph, A_exc, iQ1, iQ2):
    nQl = iQ2 - iQ1
    log.debug("\t rank: " + str(mpi.rank) + " -> " + str(nQl))
    G_cc = np.zeros((nQl, p.N_q, p.alpha, p.beta, p.nmodes), dtype='complex64')
    G_vv = np.zeros((nQl, p.N_q, p.alpha, p.beta, p.nmodes), dtype='complex64')

    for Q_ind, q_ind, k_ind in itertools.product(range(iQ1, iQ2), range(p.N_q), range(p.N_k)):
        Qq_ind = k2ik(ik2k(Q_ind)+ik2k(q_ind))
        kq_ind = k2ik(ik2k(k_ind)+ik2k(q_ind))
        kQ_ind = k2ik(ik2k(k_ind)-ik2k(Q_ind))
        kQq_ind= k2ik(ik2k(k_ind)-ik2k(Q_ind)-ik2k(q_ind))
        #if mpi.rank == MPI_ROOT:
        #    log.debug(f"\t kq_ind {kq_ind} - Qq_ind {Qq_ind} - kQ_ind {kQ_ind} - kQq_ind {kQq_ind}")

        if not (0 <= Q_ind - iQ1 < nQl):
            log.error(f"Q_ind {Q_ind} - iQ1 {iQ1} out of bounds")

        G_cc[Q_ind-iQ1, q_ind, :, :, :] += np.einsum(
            'mij,nik,jkl->nml',
            A_exc[Qq_ind, 0:p.beta, kq_ind, :, :].conj(),
            A_exc[Q_ind, 0:p.alpha, k_ind, :, :],
            g_elph[q_ind, kq_ind, p.N_v:p.N_v+p.N_c, p.N_v:p.N_v+p.N_c, :].conj()
        )

        G_vv[Q_ind-iQ1, q_ind, :, :, :] += -np.einsum(
            'mij,nkj,kil->nml',
            A_exc[Qq_ind, 0:p.beta, k_ind, :, :].conj(),
            A_exc[Q_ind, 0:p.alpha, k_ind, :, :],
            g_elph[q_ind, kQ_ind, 0:p.N_v, 0:p.N_v, :].conj()
        )
    
    return G_cc, G_vv