import numpy as np
import yaml
import logging
from logmod import log_class
from logmod import colored_log_class
from read_input_data import read_input_data

def main():
	log = configure()
	log.debug(" ++++++++++++++++++++++++++++++  START READING INPUT DATA ++++++++++++++++++++++++++++++ ")
	read_input_data(log)

def configure():
	# open config.yml
	try:
		f = open("./config.yml")
	except:
		raise Exception("config.yml cannot be opened")
	config = yaml.load(f, Loader=yaml.Loader)
	f.close()
	
	if 'LOG_LEVEL' in config:
		if config['LOG_LEVEL'] == "DEBUG":
			LOG_LEVEL = logging.DEBUG
		elif config['LOG_LEVEL'] == "INFO":
			LOG_LEVEL = logging.INFO
		elif config['LOG_LEVEL'] == "WARNING":
			LOG_LEVEL = logging.WARNING
		elif config['LOG_LEVEL'] == "ERROR":
			LOG_LEVEL = logging.ERROR
		elif config['LOG_LEVEL'] == "CRITICAL":
			LOG_LEVEL = logging.CRITICAL
		else:
			LOG_LEVEL = logging.NOTSET
	if 'COLORED_LOGGING' in config:
		COLOR = config['COLORED_LOGGING']
	if 'LOGFILE' in config:
		LOGFILE = config['LOGFILE']
	# set up logging system
	if COLOR:
		log = colored_log_class(LOG_LEVEL)
	else:
		log = log_class(LOG_LEVEL, LOGFILE)
	return log
	
if __name__ == '__main__':
	main()
