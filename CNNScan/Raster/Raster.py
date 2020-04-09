import typing

import numpy

from CNNScan.Ballot import BallotDefinitions, MarkedBallots
import CNNScan.Mark

"""
(Unimplemented) helper class to apply "marks" to regions on a contest.
"""
class Rasterizer:
	def __init__(self, ballot:BallotDefinitions.Ballot, mark_database=None):
		self.ballot = ballot
		self.mark_database = mark_database

	def rasterize_marked(self, marked_ballot:MarkedBallots.MarkedBallot, mark=None):
		if mark is None:
			# If no specific mark is required, then pick one from the database at random.
			mark = self.mark_database.get_random_mark()



# Take in a ballot, which contains a PDF file of a ballot. Coordinates are %'s relative to the size of the ballot PDF.
# All coordinates of the returned ballot should be absolute (i.e. in pixels) rather than percentages
def rasterize_ballot_template(ballot : BallotDefinitions.Ballot, directory : str, dpi:int = 400) -> BallotDefinitions.Ballot:
	# Establish pre-conditions that ballots have relative coordinates.
	for contest in ballot.contests:
		assert isinstance(contest.bounding_rect, CNNScan.Ballot.Positions.RelativePosition)
		for option in contest.options:
			assert isinstance(option.bounding_rect, CNNScan.Ballot.Positions.RelativePosition)
	# TODO: Convert ballot.ballot_file from a PDF to a PNG, stored in directory. Render at the correct DPI.
	converted_contests = []
	for contest in ballot.contests:
		# TODO: Figure out how to rasterize an individual contest to a properly size PNG
		(subimage, contest_pos) = rasterize_contest(contest, numpy.ndarray((0,0)))
		# TODO: Convert options from relative to absolute coordinates.
		converted_options = []
		for option in contest.options:
			pass
		new_contest = BallotDefinitions.Contest(contest.index, contest.contest_name, contest.description, converted_options, contest_pos)
		# TODO: Determine where contest image is saved.
		new_contest.file = "??"
		new_contest.image = subimage
	ret_val = BallotDefinitions.Ballot(converted_contests, ballot.ballot_file)

	# Assert postconditions that all positions are now absolute, and that each contest has an image.
	for contest in ballot.contests:
		assert isinstance(contest.bounding_rect, CNNScan.Ballot.Positions.PixelPosition)
		assert contest.image is not None
		for option in contest.options:
			assert isinstance(option.bounding_rect, CNNScan.Ballot.Positions.PixelPosition)
	return ret_val

# TODO: Convert a relative-positioned contest to an 
def rasterize_contest(Contest : BallotDefinitions.Contest, ballot_image : numpy.ndarray) -> typing.Tuple[numpy.ndarray, CNNScan.Ballot.Positions.PixelPosition]:
	return numpy.ndarray((0,0)), CNNScan.Ballot.Positions.PixelPosition()
	pass