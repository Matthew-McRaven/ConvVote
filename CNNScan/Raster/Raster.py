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



# Take in a ballot, which contains a PDF file of a ballot. Coordinates are %'s relative to the size of the
# All coordinates of the returned ballot should be absolute (i.e. in pixels) rather than percentages
def rasterize_ballot_template(ballot : BallotDefinitions.Ballot, directory : str, dpi:int = 400) -> BallotDefinitions.Ballot:
	pass

def rasterize_contest():
	pass