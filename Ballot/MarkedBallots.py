from . import BallotDefinitions
# Classes corresponding to a specific individual's marked ballot.
class MarkedContest:
	def __init__(self, contest=None, image=None, actual_vote_index=[]):
		self.contest = contest
		self.image = image
		self.actual_vote_index = actual_vote_index
		self.computed_vote_index = [int()]

class MarkedBallot:
	def __init__(self, ballot, marked_contests):
		self.ballot_def = ballot
		self.marked_contest = marked_contests