import typing
from dataclasses import dataclass
from . import Positions

# Classes describing what contests may be found on a ballot,
# where the contest is located on the ballot, and what options
# are available for each contest.
class Option:
	def __init__(self, index=0, description="", rel_bounding_rect=('x1', 'y1', 'x2', 'y2')):
		self.index = index
		self.description = description
		assert isinstance(rel_bounding_rect, Positions.RelativePosition) or isinstance(rel_bounding_rect, Positions.PixelPosition)
		self.rel_bounding_rect = rel_bounding_rect
		self.abs_bounding_rect = None

class Contest:
	def __init__(self, index=0, name="", description="", options=[Option() for i in range(0)],
	             rel_bounding_rect=Positions.PixelPosition(Positions.PixelPoint(0,0), Positions.PixelPoint(1,1)), contest_file=""):
		self.index = index
		self.contest_name = name
		self.description = description
		self.options = options
		assert isinstance(rel_bounding_rect, Positions.RelativePosition) or isinstance(rel_bounding_rect, Positions.PixelPosition)
		# Bounding
		self.rel_bounding_rect = rel_bounding_rect
		self.abs_bounding_rect = None
		
		# Assert that option is contained entirely within a contest.
		for option in self.options:
			assert self.rel_bounding_rect.upper_left.x <= option.rel_bounding_rect.upper_left.x
			assert self.rel_bounding_rect.upper_left.y <= option.rel_bounding_rect.upper_left.y
			assert self.rel_bounding_rect.lower_right.x >= option.rel_bounding_rect.lower_right.x
			assert self.rel_bounding_rect.lower_right.x >= option.rel_bounding_rect.lower_right.x

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
		self._contests = 0
		self._options_len = 0


	def Ballot(self, *args, **kwargs):
		new_ballot = Ballot(*args, ballot_index=len(self.ballots), **kwargs)
		self.ballots.append(new_ballot)
		self._contests = sum([len(ballot.contests) for ballot in self.ballots])
		self._options_len = max([max([len(contest.options) for contest in ballot.contests]) for ballot in self.ballots])
		return new_ballot
	
	def num_contests(self):
		return self._contests 
	
	def max_options(self):
		return self._options_len

	def __len__(self):
		return len(self.ballots)
