from logmod import log
from read_input_data import read_data
from input_parser import parser
from common.param import p
from exc_ph_mod import compute_excph
from mpi_module import mpi
from read_io import save_Geph
import os

def main():
	yml_input = parser.parse_args().yml_inp[0]
	if mpi.rank == mpi.root:
		log.info("\t input file: " + yml_input)
		log.debug("\t ++++++++++++++++++++++++++++++  START READING INPUT DATA ++++++++++++++++++++++++++++++ ")
	p.read_input_parameters(yml_input)
	if mpi.rank == mpi.root:
		log.debug("\t data dir " + p.data_dir)
		if not os.path.exists(p.output_data_dir):
			os.mkdir(p.output_data_dir)
	A_exc, exc_freq, g_elph, ph_freq = read_data()
	if mpi.rank == mpi.root:
		log.info("\n")
		log.info("\t ---------------------------------------------------------------------------------------- ")
		log.info("\t DATA COLLECTED")
		log.info("\t ---------------------------------------------------------------------------------------- ")
		log.info("\n")
		log.info("\t ---------------------------------------------------------------------------------------- ")
		log.info("\t COMPUTE EXCITON-PHONON COUPLING")
		log.info("\t ---------------------------------------------------------------------------------------- ")
	iQ_lim = mpi.split_range(p.N_Q)
	[iQ1, iQ2] = iQ_lim[mpi.rank]
	log.debug("\t rank: " + str(mpi.rank) + " - " + str(iQ1) + " - " + str(iQ2))
	G_cc, G_vv = compute_excph(g_elph, A_exc, iQ1, iQ2)
	# combine two contributions
	G = G_cc + G_vv
	# save Geph to file
	dir_path = p.output_data_dir+'/excph_dir'
	if mpi.rank == mpi.root:
		if not os.path.isdir(dir_path):
			os.mkdir(dir_path)
	mpi.comm.Barrier()
	save_Geph(G, iQ1, iQ2, dir_path)
	mpi.comm.Barrier()
	if mpi.rank == mpi.root:
		log.info("\n")
		log.info("\t ---------------------------------------------------------------------------------------- ")
		log.info("\t EXCITON-PHONON COUPLING DATA SAVED")
		log.info("\t ---------------------------------------------------------------------------------------- ")
		log.info("\n")

	
if __name__ == '__main__':
	main()
