import numpy as np

class Mark:
	def __init__(self, mark_file):
		self.mark_file = mark_file
	
	def rasterize(self,)->np.ndarray:
		return np.ndarray((10,10))

class MarkDatabase:
	def __init__(self, database_file = None):
		pass

	def get_random_mark(self):
		pass