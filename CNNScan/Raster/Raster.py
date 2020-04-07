from CNNScan.Ballot import BallotDefinitions, MarkedBallots
import CNNScan.Raster.Mark
class Rasterizer:
	def __init__(self, ballot:BallotDefinitions.Ballot, mark_database=None):
		self.ballot = ballot
		self.mark_database = mark_database

	def rasterize_marked(self, marked_ballot:MarkedBallots.MarkedBallot, mark=None):
		if mark is None:
			# If no specific mark is required, then pick one from the database at random.
			mark = self.mark_database.get_random_mark()
		