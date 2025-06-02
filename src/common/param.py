#!/usr/bin/env python3
from __future__ import print_function, division
import numpy as np
import yaml
from logmod import log
from common.constants import Ha2eV


class parameters:
    def __init__(self):

        # ─── File Paths ──────────────────────────────────────────────────────────
        self.data_dir       = "/Users/keyneshdongol/Downloads/excph-devel(1)/yambo-qe-nk662-exp-lat/QPT6/dvscf/bn.save"
        self.path_bse_data  = "/Users/keyneshdongol/Downloads/excph-devel(1)/yambo-qe-nk662-exp-lat/exciton/SAVE"
        self.elph_dir       = "/Users/keyneshdongol/Downloads/excph-devel(1)/yambo-qe-nk662-exp-lat/QPT6/dvscf/bn.save/SAVE"
        self.path_elph_data = self.elph_dir + "/ndb.elph_gkkp_expanded_fragment_"
        # ─────────────────────────────────────────────────────────────────────────

        # Exciton Band Parameters (leave α,β as you like)
        self.alpha = 8
        self.beta  = 8

        # Coarse Q-mesh
        Qmesh = [8, 8, 2]
        self.qmesh = Qmesh.copy()
        self.N_Q   = np.prod(Qmesh)
        self.N_q   = self.N_Q

        # Fine Q-mesh (if used)
        fQmesh = [8, 8, 2]
        self.fqmesh = fQmesh.copy()
        self.fN_Q   = np.prod(fQmesh)
        self.fN_q   = self.fN_Q

        # Electronic Parameters
        self.N_Q    = 72
        self.N_k    = 72
        self.N_v    = 2
        self.N_c    = 2
        self.beta   = 8
        self.nbnds = 10                # # bands in el-ph calculation

        # phonon modes
        self.nmodes = 12               # set this to match your NetCDFs

    def read_input_parameters(self, yml_input):
        log.info("\t reading input file: " + yml_input)
        try:
            f = open(yml_input)
        except:
            msg = "\t COULD NOT FIND: " + yml_input
            log.error(msg)
            return
        inp = yaml.load(f, Loader=yaml.Loader)
        f.close()
        # working dir
        if 'working_dir' in inp:
            self.working_dir = inp['working_dir']
        log.debug("\t working dir: " + self.working_dir)
        # data dir
        if 'data_dir' in inp:
            self.data_dir = self.working_dir + '/' + inp['data_dir']
        if 'output_data_dir' in inp:
            self.output_data_dir = self.working_dir + '/' + inp['output_data_dir']
        # el-ph dir
        if 'elph_dir' in inp:
            self.elph_dir = self.working_dir + '/' + inp['elph_dir']
        # bands in el-ph calculation
        if 'nbnds' in inp:
            self.nbnds = inp['nbnds']
        # n. conduction bands
        if 'N_c' in inp:
            self.N_c = inp['N_c']
        # n. valence bands
        if 'N_v' in inp:
            self.N_v = inp['N_v']
        # n. electrons
        if 'N_e' in inp:
            self.N_e = inp['N_e']
        # exciton bands
        if 'alpha' in inp:
            self.alpha = inp['alpha']
        if 'beta' in inp:
            self.beta = inp['beta']
        # Q mesh
        if 'Qmesh' in inp:
            self.qmesh = inp['Qmesh']
            self.N_Q = np.prod(self.qmesh)
            self.N_q = self.N_Q
            self.N_k = self.N_Q
        # elph data
        self.path_elph_data = self.elph_dir + '/ndb.elph_gkkp_expanded_fragment_'


p = parameters()

# These “path_…” variables allow your read_io script to refer to them:
path_excph_data = "../dvscf/bn.save/SAVE/ndb.excph_gkkp_fragment_"
path_ex         = "../dvscf/bn.save/SAVE/"
path_bse        = "./"   # if your read_io_mpi.py does something like `path_bse + f"ndb.BS_diago_Q{…}"`
path_excph      = "./"

# Phonon Mode Parameters
nmodes = 12  # Number of phonon modes

# Reciprocal Lattice Vectors
b = [[1.000000, 0.577350, 0.000000],  # Reciprocal lattice vectors for the structure
     [0.000000, 1.154701, 0.000000],
     [0.000000, 0.000000, 0.326020]]

b_iku = [[1.000000, 0.500000, 0.000000],  # Reciprocal lattice vectors used for indexing
         [0.000000, 1.000000, 0.000000],
         [0.000000, 0.000000, 1.000000]]

# Unit Cell Volume (hBN structure)
vol_hbn = 283.8914  # Unit cell volume in atomic units
vol = vol_hbn       # Alias for volume, may change depending on material

# Temperature Parameters
T_0_ph = 77.         # Experimental phonon temperature (K)
T_0_exc = 8.        # Experimental excitation temperature (K)
T_room = 300.       # Room temperature (K)
fac_T_eff = 1.      # Temperature scaling factor (if needed)

# Dielectric Constants
epsilxx = 6.9       # Dielectric constant along the x-axis
epsilzz = 3.5       # Dielectric constant along the z-axis

# Max Residual Energies (eV) for different directions
MAX_RES_xx = 0.30853E+01
MAX_RES_yy = 0.30853E+01
MAX_RES_zz = 0.90425E+00

# Broadening and Scattering Parameters (in eV)
sig_pl = 0.004 / Ha2eV  # Plasmon broadening
sig_scat = 0.004 / Ha2eV  # Scattering broadening

# Energy Range for Plotting (in eV)
e1 = 5.00 / Ha2eV
e2 = 5.30 / Ha2eV
energy = np.linspace(e1, e2, 4000)  # Energy range from 5.0 eV to 5.3 eV with 4000 points

# Temperature Range for Calculations
T_min = 5.        # Minimum temperature for calculations (K)
T_max = 300.      # Maximum temperature for calculations (K)
T_steps = 59.     # Number of temperature steps

# Number of Degenerate States
ndeg = 2          # Number of degenerate states (e.g., for excitons)
