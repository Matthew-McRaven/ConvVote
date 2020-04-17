import typing
import os
import math

import numpy
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from PIL import Image

import CNNScan.Reco.Load
from CNNScan.Ballot import BallotDefinitions, MarkedBallots
import CNNScan.Mark

to_pos = CNNScan.Ballot.Positions.to_pixel_pos
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


# Convert a bounding rectangle from one aspect ratio to another.
def fix_rect(object, width, height, page, old_width=1, old_height=1, width_offset=0, height_offset=0):
	x1 = round(width/old_width * (object.bounding_rect.lower_right.x - width_offset))
	y1 = round(height/old_height * (object.bounding_rect.lower_right.y - height_offset))
	x0 = round(width/old_width * (object.bounding_rect.upper_left.x - width_offset))
	y0 = round(height/old_height * (object.bounding_rect.upper_left.y - height_offset))
	#print(x0, y1, x1, y1)
	object.bounding_rect = CNNScan.Ballot.Positions.to_pixel_pos(x0, y0, x1, y1, page)

# Load the PDF associated with a ballot template, convert the PDF to a PIL.Image,
# convert bounding rectangles from %'s to pixels, and store each of the page's images in ballot.pages.
def rasterize_ballot_image(ballot : BallotDefinitions.Ballot, directory : str, dpi:int = 400):
	# Establish pre-conditions that ballots have relative coordinates.
	# print("ballot",ballot,"\ndirectory",directory)
	for contest in ballot.contests:
		assert isinstance(contest.bounding_rect, CNNScan.Ballot.Positions.RelativePosition)
		for option in contest.options:
			assert isinstance(option.bounding_rect, CNNScan.Ballot.Positions.RelativePosition)

	bf = convert_from_path(ballot.ballot_file, output_folder=directory,output_file="tmp",dpi=dpi)
	temp = os.listdir(directory)
	ballot_pages = []
	for img in temp:
		if "tmp" in img:
			ballot_pages.append(img)
	ballot_pages.sort()

	# Load all ballot pages as PNGs into memory.
	for page in ballot_pages:
		ballot.pages.append(Image.open(f"{directory}/{page}").convert("RGBA"))

	# Adjusted contest, option coordinates from %'s to pixels.
	for contest in ballot.contests:
		page = contest.bounding_rect.page
		width,height = ballot.pages[page].size
		#print(option.bounding_rect)
		fix_rect(contest, width, height, page)
		#print(contest.bounding_rect)
		for option in contest.options:
			#print(option.bounding_rect)
			fix_rect(option, width, height, page)
			#print(option.bounding_rect)
	return ballot

# TODO: 
# Return a new ballot template where contests, bounding 
def crop_template(ballot_def : BallotDefinitions.Ballot):
	converted_contests = []
	# Assert postconditions that all positions are now absolute, and that each contest has an image.
	for contest in ballot_def.contest:
		page = contest.bounding_rect.page
		width,height = ballot_def.pages[page].size

		
		x0, y0 = contest.bounding_rect.upper_left.x, contest.bounding_rect.upper_left.y
		x1, y1 = contest.bounding_rect.lower_right.x, contest.bounding_rect.lower_right.y
		#print(option.bounding_rect)
		raise NotImplementedError("Have not converted ballot templates yet.")
		# TODO: Fix bounding rectangles on contests
		#fix_rect(marked_contest, 1, 1, page, x0, y0)
		#print(contest.bounding_rect)
		# TODO: Fix bounding rectangles on options
		#for option in marked_contest.options:
			#print(option.bounding_rect)
			#fix_rect(option, 1, 1, page, x0, y0)
			#print(option.bounding_rect)

	ret_val = BallotDefinitions.Ballot(converted_contests, ballot_def.ballot_file)
	ret_val.pages = ballot_def.pages

	for contest in ret_val.contests:
		assert isinstance(contest.bounding_rect, CNNScan.Ballot.Positions.PixelPosition)
		assert contest.image is not None
		for option in contest.options:
			assert isinstance(option.bounding_rect, CNNScan.Ballot.Positions.PixelPosition)

	return ret_val

def crop_contests(ballot_def : BallotDefinitions.Ballot, marked_ballot : MarkedBallots.MarkedBallot) -> MarkedBallots.MarkedBallot:
	# Require that the marked ballot is already marked
	assert marked_ballot.pages is not None
	#converted_contests = []
	# TODO: Create new ballot definition rather than update in place
	# Adjusted contest, option coordinates from %'s to pixels.
	for marked_contest in marked_ballot.marked_contest:
		index = marked_contest.index
		page = ballot_def.contests[index].bounding_rect.page
		width,height = marked_ballot.pages[page].size

		bounding_rect = ballot_def.contests[marked_contest.index].bounding_rect
		x0, y0 = bounding_rect.upper_left.x, bounding_rect.upper_left.y
		x1, y1 = bounding_rect.lower_right.x, bounding_rect.lower_right.y
		#print(option.bounding_rect)
		marked_contest.image = marked_ballot.pages[page].crop((x0, y0, x1, y1))
		# TODO: Fix bounding rectangles on contests
		#fix_rect(marked_contest, 1, 1, page, x0, y0)
		#print(contest.bounding_rect)
		# TODO: Fix bounding rectangles on options
		#for option in marked_contest.options:
			#print(option.bounding_rect)
			#fix_rect(option, 1, 1, page, x0, y0)
			#print(option.bounding_rect)
