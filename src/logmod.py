import logging
import inspect
import sys
from colorlog import ColoredFormatter
import yaml
import time  # Added for timing
from functools import wraps # Added for decorator best practices

class log_class:
	def __init__(self, LOG_LEVEL, logfile):
		format = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s"
		file_formatter = logging.Formatter(format)
		file_handler = logging.FileHandler(logfile)
		file_handler.setFormatter(file_formatter)
		self.logger  = logging.getLogger()
		self.logger.addHandler(file_handler)
		self.logger.setLevel(LOG_LEVEL)
		self.level = self.logger.getEffectiveLevel()
		self.msg_len_min = 58
	@staticmethod
	def __get_call():
		stack = inspect.stack()
		# stack[1] gives previous function ('info' in our case)
		# stack[2] gives before previous function and so on
		fn = stack[2][1].split('/')
		ln = stack[2][2]
		func = stack[2][3]
		return fn[-1], func, ln
	def info(self, message, *args):
		msg = "{} - {} at line {} : {}"
		msg = msg.format(*self.__get_call(), f"{message : <30}").split(':')
		msg2 = msg[0] + ":"
		if len(msg[0]) < self.msg_len_min:
			for i in range(self.msg_len_min - len(msg[0])):
				msg2 += " "
		msg2 += message
		self.logger.info(msg2, *args)
	def debug(self, message, *args):
		msg = "{} - {} at line {} : {}"
		msg = msg.format(*self.__get_call(), f"{message : <30}").split(':')
		msg2 = msg[0] + ":"
		if len(msg[0]) < self.msg_len_min:
			for i in range(self.msg_len_min - len(msg[0])):
				msg2 += " "
		msg2 += message
		self.logger.debug(msg2, *args)
	def warning(self, message, *args):
		msg = "{} - {} at line {} : {}"
		msg = msg.format(*self.__get_call(), f"{message : <30}").split(':')
		msg2 = msg[0] + ":"
		if len(msg[0]) < self.msg_len_min:
			for i in range(self.msg_len_min - len(msg[0])):
				msg2 += " "
		msg2 += message
		self.logger.warning(msg2, *args)
	def error(self, message, *args):
		msg = "{} - {} at line {} : {}"
		msg = msg.format(*self.__get_call(), f"{message : <30}").split(':')
		msg2 = msg[0] + ":"
		if len(msg[0]) < self.msg_len_min:
			for i in range(self.msg_len_min - len(msg[0])):
				msg2 += " "
		msg2 += message
		self.logger.error(msg2, *args)
		sys.exit(1)

	def time_this(self, func_to_time):
		"""Decorator to time a function and log its execution time using self.logger.info."""
		@wraps(func_to_time)
		def wrapper(*args, **kwargs):
			start_time = time.perf_counter()
			result = func_to_time(*args, **kwargs)
			end_time = time.perf_counter()
			duration = end_time - start_time
			# Using self.logger.info directly for a distinct timing message format
			self.logger.info(f"[TIMING] {func_to_time.__module__}.{func_to_time.__name__} executed in {duration:.4f}s")
			return result
		return wrapper

class colored_log_class:
	def __init__(self, LOG_LEVEL):
		LOG_FORMAT= "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)4s"
		colors={
			'DEBUG':    'blue,bg_white',
			'INFO':     'green',
			'WARNING':  'yellow',
			'ERROR':    'black,bg_red',
			'CRITICAL': 'red,bg_white',
		}
		logging.root.setLevel(LOG_LEVEL)
		formatter = ColoredFormatter(LOG_FORMAT, log_colors=colors)
		stream = logging.StreamHandler()
		stream.setLevel(LOG_LEVEL)
		stream.setFormatter(formatter)
		self.log = logging.getLogger('pythonConfig')
		self.log.setLevel(LOG_LEVEL)
		self.log.addHandler(stream)
		self.level = self.log.getEffectiveLevel()
		self.msg_len_min = 58
	@staticmethod
	def __get_call():
		stack = inspect.stack()
		# stack[1] gives previous function ('info' in our case)
		# stack[2] gives before previous function and so on
		fn = stack[2][1].split('/')
		ln = stack[2][2]
		func = stack[2][3]
		return fn[-1], func, ln
	def info(self, message, *args):
		msg = "{} - {} at line {} : {}"
		msg = msg.format(*self.__get_call(), f"{message : <30}").split(':')
		msg2 = msg[0] + ":"
		if len(msg[0]) < self.msg_len_min:
			for i in range(self.msg_len_min - len(msg[0])):
				msg2 += " "
		msg2 += message
		self.log.info(msg2, *args)
	def debug(self, message, *args):
		msg = "{} - {} at line {} : {}"
		msg = msg.format(*self.__get_call(), f"{message : <30}").split(':')
		msg2 = msg[0] + ":"
		if len(msg[0]) < self.msg_len_min:
			for i in range(self.msg_len_min - len(msg[0])):
				msg2 += " "
		msg2 += message
		self.log.debug(msg2, *args)
	def warning(self, message, *args):
		msg = "{} - {} at line {} : {}"
		msg = msg.format(*self.__get_call(), f"{message : <30}").split(':')
		msg2 = msg[0] + ":"
		if len(msg[0]) < self.msg_len_min:
			for i in range(self.msg_len_min - len(msg[0])):
				msg2 += " "
		msg2 += message
		self.log.warning(msg2, *args)
	def error(self, message, *args):
		msg = "{} - {} at line {} : {}"
		msg = msg.format(*self.__get_call(), f"{message : <30}").split(':')
		msg2 = msg[0] + ":"
		if len(msg[0]) < self.msg_len_min:
			for i in range(self.msg_len_min - len(msg[0])):
				msg2 += " "
		msg2 += message
		self.log.error(msg2, *args)
		sys.exit(1)

	def time_this(self, func_to_time):
		"""Decorator to time a function and log its execution time using self.log.info."""
		@wraps(func_to_time)
		def wrapper(*args, **kwargs):
			start_time = time.perf_counter()
			result = func_to_time(*args, **kwargs)
			end_time = time.perf_counter()
			duration = end_time - start_time
			# Using self.log.info directly for a distinct timing message format
			self.log.info(f"[TIMING] {func_to_time.__module__}.{func_to_time.__name__} executed in {duration:.4f}s")
			return result
		return wrapper

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

log = configure()
