from read_io import read_qpoints_yambo 

def read_input_data(log):
	log.info(" +++++++++++++++++++++++++++++  READING YAMBO Q POINTS ++++++++++++++++++++++++++++++ ")
	Q_yambo = read_qpoints_yambo()
	'''
        # Initialize arrays
	A_exc = np.zeros((N_Q, beta, N_k, N_v, N_c), dtype='complex64')
	exc_freq = np.zeros((N_Q, beta), dtype='float32')
	
	# Read BSE wavefunction data
	A_exc, exc_freq = read_bse_wavefunction(Q_yambo, A_exc, exc_freq)
	
	# Save exciton wavefunctions and energies
	A_exc.tofile('A_exc.dat')
	exc_freq.tofile('exc_freq.dat')
	
	# Initialize arrays for electron-phonon coupling and phonon frequencies
	g_elph = np.zeros((N_Q, N_k, N_v + N_c, N_v + N_c, nmodes), dtype='complex64')
	ph_freq = np.zeros((N_Q, nmodes), dtype='float32')
	
	# Read electron-phonon coupling data
	g_elph, ph_freq = read_elph_data(Q_yambo, g_elph, ph_freq)
	
	# Normalize the electron-phonon matrix
	g_elph, ph_freq = normalize_elph(g_elph, ph_freq)
	
	# Save electron-phonon coupling matrix and phonon frequencies
	g_elph.tofile('g_elph.dat')
	ph_freq.tofile('ph_freq.dat')
	'''
	print('Job done.')
