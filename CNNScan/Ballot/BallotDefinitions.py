import typing
from dataclasses import dataclass
from . import Positions

# Classes describing what contests may be found on a ballot,
# where the contest is located on the ballot, and what options
# are available for each contest.
class Option:
	def __init__(self, index=0, description="", bounding_rect=('x1', 'y1', 'x2', 'y2')):
		self.index = index
		self.description = description
		assert isinstance(bounding_rect, Positions.RelativePosition) or isinstance(bounding_rect, Positions.PixelPosition)
		self.bounding_rect = bounding_rect

class Contest:
	def __init__(self, index=0, name="", description="", options=[Option() for i in range(0)],
	             bounding_rect=Positions.PixelPosition(Positions.PixelPoint(0,0), Positions.PixelPoint(1,1)), contest_file=""):
		self.index = index
		self.contest_name = name
		self.description = description
		self.options = options
		assert isinstance(bounding_rect, Positions.RelativePosition) or isinstance(bounding_rect, Positions.PixelPosition)
		self.bounding_rect = bounding_rect
		
		# Assert that option is contained entirely within a contest.
		for option in self.options:
			assert self.bounding_rect.upper_left.x <= option.bounding_rect.upper_left.x
			assert self.bounding_rect.upper_left.y <= option.bounding_rect.upper_left.y
			assert self.bounding_rect.lower_right.x >= option.bounding_rect.lower_right.x
			assert self.bounding_rect.lower_right.x >= option.bounding_rect.lower_right.x

		self.contest_file = contest_file

	# Assert that removed attributes have been removed (for debugging purposes).
	def __getattribute__(self, attribute):
		assert attribute is not "image"
		return super(Contest, self).__getattribute__(attribute)

class Ballot:
	def __init__(self, contests=[Contest(index=i)  for i in range(0)], ballot_file="", ballot_index=-1):
		self.ballot_index = ballot_index
		self.contests = contests
		self.ballot_file = ballot_file
		# Store the images for each page of the printed PDF.
		# To be filled in during rasterization from PDF.
		self.pages = []

class BallotFactory:
	def __init__(self):
		self.ballots = []

	def Ballot(self, *args, **kwargs):
		new_ballot = Ballot(*args, ballot_index=len(self.ballots), **kwargs)
		self.ballots.append(new_ballot)
		return new_ballot
