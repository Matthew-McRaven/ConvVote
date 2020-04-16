# Classes corresponding to a specific individual's marked ballot.
class MarkedContest:
	def __init__(self, index=None, image=None, actual_vote_index=[]):
		# Do not store a pointer to the BallotDefinitions.Contest object.
		# This massively inflates object size, and unduly creates memory pressure.
		# This was discovered in the commit after 95ac0b79f1bc2083f6e270072637e72e6bc75ef8, 
		# while attempting to pickle makred contests and ballots to files.
		# Size inflation was on the order of 48Mb containing pointers verus 1Kb without.
		#self.contest = contest
		self.index = index
		self.image = image
		self.tensor = None
		self.actual_vote_index = actual_vote_index
		self.computed_vote_index = []
		
	def clear_data(self):
		self.image = None
		self.tensor = None

class MarkedBallot:
	def __init__(self, ballot, marked_contests, pages):
		# Do not store a pointer to the BallotDefinitions.Contest object.
		# This massively inflates object size, and unduly creates memory pressure.
		# This was discovered in the commit after 95ac0b79f1bc2083f6e270072637e72e6bc75ef8, 
		# while attempting to pickle makred contests and ballots to files.
		# Size inflation was on the order of 48Mb containing pointers verus 1Kb without.
		#self.ballot_def = ballot
		self.marked_contest = marked_contests
		# Store the images for each page of the printed PDF.
		# Cloned from the ballot definition, and marked up using the Marks module.
		self.pages = pages