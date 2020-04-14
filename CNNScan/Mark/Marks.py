import typing
import random

import numpy as np
import numpy.random

import CNNScan.Ballot.MarkedBallots
import CNNScan.Mark.Marks

from PIL import Image, ImageDraw
import PIL.ImageOps 

class MarkDatabase:
	def __init__(self):
		self.marks = []
		pass

	def insert_mark(self, mark):
		self.marks.append(mark)

	def get_random_mark(self):
		return self.marks[random.randint(0, len(self.marks)-1)]

class XMark:
	def generate(self, image: Image, mark_shape: CNNScan.Ballot.Positions.PixelPosition) -> Image:
		im = Image.new("RGBA",size=mark_shape.size())
		draw = ImageDraw.Draw(im)
		draw.line((0, 0) + im.size, fill=(255,0,0), width=4)
		draw.line((0, im.size[1], im.size[0], 0), fill=(0,255,0), width=4)
		return im

	def __call__(self, *args, **kwargs):
		return self.generate(*args, **kwargs)


class BoxMark:
	def generate(self, image: Image, mark_shape: CNNScan.Ballot.Positions.PixelPosition) -> Image:
		#print(mark_shape)
		im = Image.new("RGBA",size=mark_shape.size())
		draw = ImageDraw.Draw(im)
		draw.rectangle((0, 0) + im.size, fill=(0, 0, 0), width=4)
		return im

	def __call__(self, *args, **kwargs):
		return self.generate(*args, **kwargs)

class InvertMark:
	
	def generate(self, image: Image, mark_shape: CNNScan.Ballot.Positions.PixelPosition) -> Image:
		ul, lr = mark_shape.upper_left, mark_shape.lower_right
		as_rgb = image.crop(box=(ul.x,ul.y,lr.x,lr.y)).convert(mode="RGB")
		return PIL.ImageOps.invert(as_rgb).convert("RGBA")

	def __call__(self, *args, **kwargs):
		return self.generate(*args, **kwargs)

# TODO: Refactor application classes so that they can re-use looping constructs.
# Take a mark, and apply random noise to the mark before directly applying it to the image.
class NoisyApply:
	def __init__(self, noise:float):
		self.noise = noise

	def apply_mark(self, image, position, mark, **kwargs):
		mark_im = mark.generate(image, position)
		# Switch between row-major and column-major matrix orderings.
		noise_array = np.transpose(np.random.random(position.size()))
		noise_im = Image.fromarray(noise_array, mode="RGBA")
		mark_im = Image.blend(mark_im,noise_im, self.noise)
		ul, lr = position.upper_left, position.lower_right
		image.paste(mark_im, box=(ul.x,ul.y,lr.x,lr.y),mask=mark_im)
				
	def __call__(self, *args, **kwargs):
		return self.apply_mark(*args, **kwargs)

# Apply a mark by directly assigning 
class AssignApply:
	def apply_mark(self, image, position, mark, **kwargs):
		mark_im = mark.generate(image, position)
		ul, lr = position.upper_left, position.lower_right
		image.paste(mark_im, box=(ul.x,ul.y,lr.x,lr.y),mask=mark_im)

	def __call__(self, *args, **kwargs):
		return self.apply_mark(*args, **kwargs)


def apply_marks(contest, marked: CNNScan.Ballot.MarkedBallots.MarkedContest, mark, apply=AssignApply()):
	for which in marked.actual_vote_index:
		apply(marked.image, contest.options[which].bounding_rect, mark)
	return marked

def mark_dataset(ballot:CNNScan.Ballot.BallotDefinitions.Ballot, count=100, transform=None):
	mark_db = CNNScan.Mark.MarkDatabase()
	mark_db.insert_mark(CNNScan.Mark.BoxMark())
	return CNNScan.Reco.Load.ImageDataSet(ballot, mark_db, count, transform=transform)
