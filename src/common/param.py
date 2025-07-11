#!/usr/bin/env python3
from __future__ import print_function, division
import numpy as np
import os
import yaml
from logmod import log
from common.constants import Ha2eV, eV


class parameters:
    def __init__(self):

        # ─── File Paths ──────────────────────────────────────────────────────────
        self.data_dir       = "/Users/keyneshdongol/Downloads/excph-devel(1)/yambo-qe-nk662-exp-lat/QPT6/dvscf/bn.save"
        self.path_bse_data  = "/Users/keyneshdongol/Downloads/excph-devel(1)/yambo-qe-nk662-exp-lat/exciton/SAVE"
        self.elph_dir       = "/Users/keyneshdongol/Downloads/excph-devel(1)/yambo-qe-nk662-exp-lat/QPT6/dvscf/bn.save/SAVE"
        self.path_elph_data = self.elph_dir + "/ndb.elph_gkkp_expanded_fragment_"
        # ─────────────────────────────────────────────────────────────────────────

        # Exciton Band Parameters
        self.alpha = 8
        self.beta  = 8

        # Coarse Q-mesh (updated to match working parameters)
        Qmesh = [6, 6, 2]
        self.qmesh = Qmesh.copy()
        self.N_Q   = np.prod(Qmesh)
        self.N_q   = self.N_Q

        # Fine Q-mesh (updated to match working parameters)
        fQmesh = [6, 6, 2]
        self.fqmesh = fQmesh.copy()
        self.fN_Q   = np.prod(fQmesh)
        self.fN_q   = self.fN_Q

        # Electronic Parameters
        self.N_k    = self.N_Q
        self.N_v    = 2
        self.N_c    = 2
        self.N_e    = 8                # Updated to match working parameters (total electrons)
        self.nbnds = 10                # # bands in el-ph calculation

        # phonon modes
        self.nmodes = 12               # set this to match your NetCDFs

    def read_input_parameters(self, yml_input):
        log.info("\\t reading input file: " + yml_input)
        try:
            f = open(yml_input) # This is where the path is used
        except:
            msg = "\\t COULD NOT FIND: " + yml_input
            log.error(msg)
            return
        inp = yaml.load(f, Loader=yaml.Loader)
        f.close()
        # working dir
        if 'working_dir' in inp:
            self.working_dir = inp['working_dir']
        # Ensure self.working_dir has a default if not in YAML, or ensure YAML always provides it.
        # For this edit, we assume self.working_dir is set (e.g., from YAML or a default in __init__).
        # If __init__ doesn't set it and YAML doesn't, getattr would be safer for logging.
        log.debug("\\t working dir: " + getattr(self, 'working_dir', 'NOT SET (defaulting to current for joining if relative)'))

        # data dir
        if 'data_dir' in inp:
            input_data_path = inp['data_dir']
            if os.path.isabs(input_data_path):
                self.data_dir = input_data_path
            else:
                # Use current working directory if self.working_dir is not meaningfully set
                base_dir_for_join = getattr(self, 'working_dir', '.') if self.working_dir else '.'
                self.data_dir = os.path.join(base_dir_for_join, input_data_path)

        # output_data_dir
        if 'output_data_dir' in inp:
            # output_data_dir is usually relative to working_dir
            base_dir_for_join = getattr(self, 'working_dir', '.') if self.working_dir else '.'
            self.output_data_dir = os.path.join(base_dir_for_join, inp['output_data_dir'])

        # el-ph dir
        if 'elph_dir' in inp:
            input_elph_path = inp['elph_dir']
            if os.path.isabs(input_elph_path):
                self.elph_dir = input_elph_path
            else:
                base_dir_for_join = getattr(self, 'working_dir', '.') if self.working_dir else '.'
                self.elph_dir = os.path.join(base_dir_for_join, input_elph_path)
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
        else:
            self.N_e = 8  # Default value matching working script
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
        # elph data path (this should use the potentially updated self.elph_dir)
        # Ensure self.elph_dir is defined before using it here. It's set if 'elph_dir' in inp, otherwise uses __init__ default.
        self.path_elph_data = os.path.join(self.elph_dir, 'ndb.elph_gkkp_expanded_fragment_')

        # bse_data_dir
        if 'bse_data_dir' in inp:
            input_bse_path = inp['bse_data_dir']
            if os.path.isabs(input_bse_path):
                self.path_bse_data = input_bse_path
            else:
                base_dir_for_join = getattr(self, 'working_dir', '.') if self.working_dir else '.'
                self.path_bse_data = os.path.join(base_dir_for_join, input_bse_path)


p = parameters()

# Additional paths for compatibility
path_elph_data = p.path_elph_data
path_excph_data = '/..'  # might need to check this later
path_ex = '/Users/keyneshdongol/Downloads/excph-devel(1)/yambo-qe-nk662-exp-lat/exciton/SAVE/'
path_bse = '/Users/keyneshdongol/Downloads/excph-devel(1)/yambo-qe-nk662-exp-lat/exciton/'
path_excph = './'

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

# Energy Range for Plotting (in eV) - Updated to match working parameters
e1 = 5.00 / Ha2eV
e2 = 6.80 / Ha2eV
energy = np.linspace(e1, e2, 4000)  # Energy range from 5.0 eV to 6.8 eV with 4000 points

# Temperature Range for Calculations
T_min = 5.        # Minimum temperature for calculations (K)
T_max = 300.      # Maximum temperature for calculations (K)
T_steps = 59.     # Number of temperature steps

# Number of Degenerate States
ndeg = 2          # Number of degenerate states (e.g., for excitons)

# Working parameters - these match the confirmed working script
# Updated Q-mesh to 6x6x2 = 72 Q-points
Qmesh = [6, 6, 2]
qmesh = Qmesh.copy()
N_Q = np.prod(Qmesh)
N_q = N_Q
fQmesh = [6, 6, 2]
fqmesh = fQmesh.copy()
fN_Q = np.prod(fQmesh)
fN_q = fN_Q

# Electronic parameters - matching working script
N_e = 8
N_k = N_Q
N_v = 2
N_c = 2
nbnds = 10
alpha = 8
beta = 8
