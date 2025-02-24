from logmod import log
from read_input_data import read_data
from input_parser import parser
from common.param import p

def main():
	yml_input = parser.parse_args().yml_inp[0]
	log.info("\t input file: " + yml_input)
	log.debug("\t ++++++++++++++++++++++++++++++  START READING INPUT DATA ++++++++++++++++++++++++++++++ ")
	p.read_input_parameters(yml_input)
	log.debug("\t data dir " + p.data_dir)
	read_data()
	
if __name__ == '__main__':
	main()
