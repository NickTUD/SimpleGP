import traceback

import jsonlines as jsonl

class GPLogger:

	def __init__(self, file_location):
		self.file_location = file_location

	def __enter__(self):
		self.log_file = jsonl.open(self.file_location, 'a')
		return self

	def write(self, message):
		self.log_file.write(message)

	def __exit__(self, exc_type, exc_value, tb):
		if exc_type is not None:
			traceback.print_exception(exc_type, exc_value, tb)
		# return False # uncomment to pass exception through

		return True
