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
		self.scale = "%"

class Contest:
	def __init__(self, index=0, name="", description="", options=[Option() for i in range(0)],
	             bounding_rect=Positions.PixelPosition(Positions.PixelPoint(0,0), Positions.PixelPoint(1,1)), contest_file=""):
		self.index = index
		self.contest_name = name
		self.description = description
		self.options = options
		assert isinstance(bounding_rect, Positions.RelativePosition) or isinstance(bounding_rect, Positions.PixelPosition)
		self.bounding_rect = bounding_rect 
		self.scale = "%"
		self.contest_file = contest_file
		# This image represents the unmarked contest which this class describes.
		# Loading the unmarked contest image many times is very costly, and so it should 
		# be cached in self.image on first use.
		self.image = None

class Ballot:
	def __init__(self, contests=[Contest(index=i)  for i in range(0)], ballot_file=""):
		self.contests = contests
		self.ballot_file = ballot_file