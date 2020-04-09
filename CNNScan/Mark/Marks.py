import typing
import random

import numpy as np
import numpy.random

import CNNScan.Ballot.MarkedBallots
import CNNScan.Mark.Marks

from PIL import Image, ImageDraw

class MarkDatabase:
	def __init__(self):
		self.marks = []
		pass

	def insert_mark(self, mark):
		self.marks.append(mark)

	def get_random_mark(self):
		return self.marks[random.randint(0, len(self.marks)-1)]

class XMark:
	def generate(self, image: np.ndarray, mark_shape: CNNScan.Ballot.Positions.PixelPosition) -> np.ndarray:
		im = Image.new("L",size=mark_shape.size())
		draw = ImageDraw.Draw(im)
		draw.line((0, 0) + im.size, fill=128)
		draw.line((0, im.size[1], im.size[0], 0), fill=128)
		output = np.array(im)
		return output

	def __call__(self, *args, **kwargs):
		return self.generate(*args, **kwargs)


class BoxMark:
	def generate(self, image: np.ndarray, mark_shape: CNNScan.Ballot.Positions.PixelPosition) -> np.ndarray:
		output = np.ndarray(mark_shape.size()[::-1])
		output.fill(0)
		return output

	def __call__(self, *args, **kwargs):
		return self.generate(*args, **kwargs)

class InvertMark:
	
	def generate(self, image: np.ndarray, mark_shape: CNNScan.Ballot.Positions.PixelPosition) -> np.ndarray:
		imint = image[mark_shape.upper_left.y:mark_shape.lower_right.y, mark_shape.upper_left.x : mark_shape.lower_right.x]
		output = np.ndarray(mark_shape.size()[::-1])
		output.fill(255)
		return  output - imint

	def __call__(self, *args, **kwargs):
		return self.generate(*args, **kwargs)

# TODO: Refactor application classes so that they can re-use looping constructs.
# Take a mark, and apply random noise to the mark before directly applying it to the image.
class NoisyApply:
	def __init__(self, noise:float):
		self.noise = noise

	def apply_mark(self, image, position, mark, **kwargs):
		mark_array = mark.generate(image, position)
		noise_array = (self.noise) * np.random.random(position.size()) - (0)
		#print(position)
		for x in range(0, position.lower_right.x-position.upper_left.x):
			for y in range(0, position.lower_right.y-position.upper_left.y):
				image[y + position.upper_left.y][x + position.upper_left.x] = noise_array[y][x] + mark_array[y][x]
				
	def __call__(self, *args, **kwargs):
		return self.apply_mark(*args, **kwargs)

# Apply a mark by directly assigning 
class AssignApply:
	def apply_mark(self, image, position, mark, **kwargs):
		mark_array = mark.generate(image, position)
		for x in range(0, position.lower_right.x-position.upper_left.x):
			for y in range(0, position.lower_right.y-position.upper_left.y):
				image[y + position.upper_left.y][x + position.upper_left.x] = mark_array[y][x]

	def __call__(self, *args, **kwargs):
		return self.apply_mark(*args, **kwargs)


def apply_marks(marked: CNNScan.Ballot.MarkedBallots.MarkedContest, mark, apply=AssignApply()):
	for which in marked.actual_vote_index:
		apply(marked.image, marked.contest.options[which].bounding_rect, mark)
	return marked