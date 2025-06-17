from read_io import read_qpoints_yambo, read_bse_wavefunction, read_elph_data, normalize_elph
from logmod import log
from common.param import p
from mpi_module import mpi, MPI_ROOT
import numpy as np
import os

@log.time_this
def read_data():
    if mpi.rank == MPI_ROOT:
        log.info("\\t READING YAMBO Q POINTS ")
    Q_yambo = read_qpoints_yambo()

    # Initialize A_exc and exc_freq as the BSE reading part below is commented out.
    # If BSE reading is enabled, these lines can be removed if read_bse_wavefunction populates them.
    A_exc = np.zeros((p.N_Q, p.beta, p.N_k, p.N_v, p.N_c), dtype='complex64')
    exc_freq = np.zeros((p.N_Q, p.beta), dtype='float32')
    if mpi.rank == MPI_ROOT:
        log.debug("\\t A_exc size (initial): " + str(A_exc.shape))
        log.debug("\\t exc_freq size (initial): " + str(exc_freq.shape))

    '''
    # Read BSE wavefunction data
    # If uncommented, the call should be:
    A_exc, exc_freq = read_bse_wavefunction(Q_yambo)

    # Save exciton wavefunctions and energies
    # A_exc.tofile('A_exc.dat')
    # exc_freq.tofile('exc_freq.dat')
    '''

    # Read or process electron-phonon coupling data
    if os.path.exists(p.output_data_dir+'/g_elph.dat') and os.path.exists(p.output_data_dir+'/ph_freq.dat'):
        if mpi.rank == MPI_ROOT:
            log.info("\\t Reading el-ph and ph. freq. data from existing .dat files...")
        g_elph = np.fromfile(p.output_data_dir+'/g_elph.dat', dtype='complex64').reshape((p.N_Q, p.N_k, p.N_v+p.N_c, p.N_v+p.N_c, p.nmodes))
        ph_freq = np.fromfile(p.output_data_dir+'/ph_freq.dat', dtype='float32').reshape((p.N_Q, p.nmodes))
    else:
        if mpi.rank == MPI_ROOT:
            log.info("\\t Reading el-ph and ph. freq. data from source NetCDF files...")
        # read_elph_data now initializes and returns arrays
        g_elph, ph_freq = read_elph_data(Q_yambo)

        if mpi.rank == MPI_ROOT:
            log.info("\\t Normalizing ELPH data...")
        g_elph, ph_freq = normalize_elph(g_elph, ph_freq)

        # Save electron-phonon coupling matrix and phonon frequencies
        if mpi.rank == MPI_ROOT:
            log.info("\\t Saving g_elph.dat and ph_freq.dat...")
            if not os.path.exists(p.output_data_dir):
                os.makedirs(p.output_data_dir, exist_ok=True)
            g_elph.tofile(p.output_data_dir + '/g_elph.dat')
            ph_freq.tofile(p.output_data_dir + '/ph_freq.dat')

    if mpi.rank == MPI_ROOT: # Log final shapes
        log.debug("\\t electron-phonon shape: " + str(g_elph.shape))
        log.debug("\\t ph. frequencies shape: " + str(ph_freq.shape))

    if mpi.rank == MPI_ROOT:
        log.info('\\t Data reading/processing job done.')

    return A_exc, exc_freq, g_elph, ph_freq
