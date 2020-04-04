class OptionDefinition:
	def __init__(self, index=0, description=""):
		self.index = index
		self.description = description

class ContestDefinition:
	def __init__(self, index=0, name="", description="", options=[OptionDefinition() for i in range(0)]):
		self.id = index
		self.contest_name = name
		self.description = description
		self.options = options

class OptionLocation:
	def __init__(self):
		self.id = int()
		self.bound_rect = ('x', 'y', 'l', 'w')

class ContestLocation:
	def __init__(self, contest_index=int(), bounding_rect=('x','y','l','w'), options=[]):
		self.id = contest_index
		self.bound_rect = bounding_rect 
		self.options = options

class BallotDefinition:
	def __init__(self):
		self.contest_definitions = [ContestDefinition(index=i)  for i in range(0)]
		self.contest_locations = [ContestLocation() for i in range(0)]
		self.ballot_file = "my_file_name"

class MarkedContest:
	def __init__(self, contest=None, image=None, actual_vote_index=0):
		self.contest = contest
		self.image = image
		self.actual_vote_index = actual_vote_index
		self.computed_vote_index = int()

class MarkedBallot:
	def __init__(self):
		self.ballot_def = BallotDefinition
		self.marked_contest = [MarkedContest() for i in range(0)]