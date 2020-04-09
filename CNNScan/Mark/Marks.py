import typing
import random

import numpy as np
import numpy.random

import CNNScan.Ballot.MarkedBallots
import CNNScan.Mark.Marks

class MarkDatabase:
	def __init__(self):
		self.marks = []
		pass

	def insert_mark(self, mark):
		self.marks.append(mark)

	def get_random_mark(self):
		return self.marks[random.randint(0, len(self.marks)-1)]

class BoxMark:
	def generate(self, image: np.ndarray, mark_shape: typing.Tuple[int, int]) -> np.ndarray:
		output = np.ndarray(mark_shape)
		output.fill(0)
		return output
	def __call__(self, *args, **kwargs):
		return self.generate(*args, **kwargs)

# Apply
class NoisyApply:
	def __init__(self, noise:float):
		self.noise = noise

	def apply_mark(self, image, position, mark, **kwargs):
		mark_array = mark.generate(image, position.size())
		noise_array = 256*(self.noise) * np.random.random(position.size()) - (0)
		#print(position)
		for x in range(0, position.lower_right.x-position.upper_left.x):
			for y in range(0, position.lower_right.y-position.upper_left.y):
				image[y + position.upper_left.y][x + position.upper_left.x] = noise_array[x][y] + mark_array[x][y]
				
	def __call__(self, *args, **kwargs):
		return apply_mark(*args, **kwargs)
		
def apply_mark(image, position, mark, **kwargs):
	mark_array = mark.generate(image, position.size())
	#print(position)
	for x in range(0, position.lower_right.x-position.upper_left.x):
		for y in range(0, position.lower_right.y-position.upper_left.y):
			image[y + position.upper_left.y][x + position.upper_left.x] = mark_array[x][y]


def apply_marks(marked: CNNScan.Ballot.MarkedBallots.MarkedContest, mark, apply=NoisyApply(.5)):
	for which in marked.actual_vote_index:
		apply(marked.image, marked.contest.options[which].bounding_rect, mark)
	return marked