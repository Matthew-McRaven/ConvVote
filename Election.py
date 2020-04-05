class Option:
	def __init__(self, index=0, description="", bounding_rect=('x1', 'y1', 'x2', 'y2')):
		self.index = index
		self.description = description
		self.bound_rect = bounding_rect

class Contest:
	def __init__(self, index=0, name="", description="", options=[Option() for i in range(0)], bounding_rect=('x1','y1','x2','y2')):
		self.id = index
		self.contest_name = name
		self.description = description
		self.options = options
		self.bound_rect = bounding_rect 

class Ballot:
	def __init__(self):
		self.contests = [Contest(index=i)  for i in range(0)]
		self.ballot_file = "my_file_name"

class MarkedContest:
	def __init__(self, contest=None, image=None, actual_vote_index=0):
		self.contest = contest
		self.image = image
		self.actual_vote_index = actual_vote_index
		self.computed_vote_index = int()

class MarkedBallot:
	def __init__(self):
		self.ballot_def = Ballot()
		self.marked_contest = [MarkedContest() for i in range(0)]