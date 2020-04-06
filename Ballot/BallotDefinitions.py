# Classes describing what contests may be found on a ballot,
# where the contest is located on the ballot, and what options
# are available for each contest.
class Option:
	def __init__(self, index=0, description="", bounding_rect=('x1', 'y1', 'x2', 'y2')):
		self.index = index
		self.description = description
		self.bounding_rect = bounding_rect

class Contest:
	def __init__(self, index=0, name="", description="", options=[Option() for i in range(0)], bounding_rect=('x1','y1','x2','y2')):
		self.index = index
		self.contest_name = name
		self.description = description
		self.options = options
		self.bounding_rect = bounding_rect 

class Ballot:
	def __init__(self, contests=[Contest(index=i)  for i in range(0)], ballot_file=""):
		self.contests = contests
		self.ballot_file = ballot_file