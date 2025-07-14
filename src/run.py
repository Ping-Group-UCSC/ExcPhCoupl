from logmod import log
from read_input_data import read_data
from input_parser import parser
from common.param import p

from mpi_module import mpi
from read_io import collect_excph_data, average_degenerate_energies
from split_excph import run_split_excph
import os

def main():
	try:
		yml_input = parser.parse_args().yml_inp[0]
		if mpi.rank == mpi.root:
			log.info("\t input file: " + yml_input)
			log.debug("\t ++++++++++++++++++++++++++++++  START READING INPUT DATA ++++++++++++++++++++++++++++++ ")
		p.read_input_parameters(yml_input)
		if mpi.rank == mpi.root:
			log.debug("\t data dir " + p.data_dir)
			if not os.path.exists(p.output_data_dir):
				os.makedirs(p.output_data_dir, exist_ok=True)

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

		# Step 2a: Compute exciton-phonon coupling fragments (split calculation)
		# Pass the loaded data arrays directly to the function.
		success = run_split_excph(g_elph, A_exc)
		if not success:
			if mpi.rank == mpi.root:
				log.error("\t Failed to compute exciton-phonon coupling fragments!")
			mpi.comm.Abort(1)

		if mpi.rank == mpi.root:
			log.info("\n")
			log.info("\t ---------------------------------------------------------------------------------------- ")
			log.info("\t EXCITON-PHONON COUPLING FRAGMENTS COMPUTED")
			log.info("\t ---------------------------------------------------------------------------------------- ")
			log.info("\n")

		if mpi.rank == mpi.root:
			log.info("\t ---------------------------------------------------------------------------------------- ")
			log.info("\t COLLECTING EXCITON-PHONON COUPLING DATA")
			log.info("\t ---------------------------------------------------------------------------------------- ")

		success = collect_excph_data()
		if not success:
			if mpi.rank == mpi.root:
				log.error("\t Failed to collect exciton-phonon coupling data!")
			mpi.comm.Abort(1)

		if mpi.rank == mpi.root:
			log.info("\n")
			log.info("\t ---------------------------------------------------------------------------------------- ")
			log.info("\t EXCITON-PHONON COUPLING DATA COLLECTED")
			log.info("\t ---------------------------------------------------------------------------------------- ")
			log.info("\n")

		if mpi.rank == mpi.root:
			log.info("\t ---------------------------------------------------------------------------------------- ")
			log.info("\t AVERAGING DEGENERATE ENERGIES")
			log.info("\t ---------------------------------------------------------------------------------------- ")

		success = average_degenerate_energies()
		if not success:
			if mpi.rank == mpi.root:
				log.error("\t Failed to average degenerate energies!")
			mpi.comm.Abort(1)

		if mpi.rank == mpi.root:
			log.info("\n")
			log.info("\t ---------------------------------------------------------------------------------------- ")
			log.info("\t DEGENERATE ENERGY AVERAGING COMPLETE")
			log.info("\t ---------------------------------------------------------------------------------------- ")
			log.info("\n")
			log.info("\t ---------------------------------------------------------------------------------------- ")
			log.info("\t")
			log.info("\t ---------------------------------------------------------------------------------------- ")
			log.info("Generated files:")
			log.info("  - A_exc.dat: Exciton wavefunctions")
			log.info("  - exc_freq.dat: Exciton frequencies")
			log.info("  - exc_freq_deg.dat: Averaged degenerate exciton frequencies")
			log.info("  - g_elph.dat: Electron-phonon coupling matrix")
			log.info("  - ph_freq.dat: Phonon frequencies")
			log.info("  - excph.dat: Complex exciton-phonon coupling matrices")
			log.info("  - excph2.dat: Squared absolute values of coupling matrices")
			log.info("\t ---------------------------------------------------------------------------------------- ")
			log.info("\t ---------------------------------------------------------------------------------------- ")
			log.info("\n")
	except Exception:
		if mpi.rank == mpi.root:
			import traceback
			log.error("Workflow failed with an unhandled exception:")
			log.error(f"{traceback.format_exc()}")
		mpi.comm.Abort(1)


if __name__ == '__main__':
	main()
