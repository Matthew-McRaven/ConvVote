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



# Take in a ballot, which contains a PDF file of a ballot. Coordinates are %'s relative to the size of the ballot PDF.
# All coordinates of the returned ballot should be absolute (i.e. in pixels) rather than percentages
def rasterize_ballot_template(ballot : BallotDefinitions.Ballot, directory : str, dpi:int = 400) -> BallotDefinitions.Ballot:
	# Establish pre-conditions that ballots have relative coordinates.
	# print("ballot",ballot,"\ndirectory",directory)
	for contest in ballot.contests:
		assert isinstance(contest.bounding_rect, CNNScan.Ballot.Positions.RelativePosition)
		for option in contest.options:
			assert isinstance(option.bounding_rect, CNNScan.Ballot.Positions.RelativePosition)

	# converty pdf of ballot and save the ballot png
	bf = convert_from_path(ballot.ballot_file, output_folder=directory,output_file="tmp")
	temp = os.listdir(directory)
	ballot_pages = []
	for img in temp:
		if "tmp" in img:
			ballot_pages.append(img)
	ballot_pages.sort()
	# ballot_pages is ordered list of individual pages of pdf ballot saved as pngs
	# print("pages:",ballot_pages)
	converted_contests = []
	cc=0
	for contest in ballot.contests:
		# TODO: Figure out how to rasterize an individual contest to a properly size PNG
		bounding = contest.bounding_rect
		ballot_png=Image.open(f"{directory}/{ballot_pages[bounding.page]}")
		width,height = ballot_png.size
		# We will be cropping the image to only contain the contest and nothing else, so we must "fix" the bounding rectanble
		contest_pos=to_pos(0, 0, round(width*(bounding.lower_right.x-bounding.upper_left.x)), round(width*(bounding.lower_right.y-bounding.upper_left.y)))
		contest_img=ballot_png.crop((width*bounding.upper_left.x,height*bounding.upper_left.y,width*bounding.lower_right.x,height*bounding.lower_right.y))
		# (subimage, contest_pos) = rasterize_contest(contest, numpy.ndarray((0,0)))
		# TODO: Convert options from relative to absolute coordinates.
		converted_options = []
		cw,ch = contest_img.size
		for option in contest.options:
			# make relative positions precise pixel locations
			# multiply relative values by height and width of this contest
			option_bounding=option.bounding_rect
			# Subtract the number of pixels "cropped off" above.
			new_x1=round(width*option_bounding.upper_left.x) - round(width*bounding.upper_left.x)
			new_y1=round(height*option_bounding.upper_left.y) - round(height*bounding.upper_left.y)
			new_x2=round(width*option_bounding.lower_right.x) - round(width*bounding.upper_left.x)
			new_y2=round(height*option_bounding.lower_right.y) - round(height*bounding.upper_left.y)
			new_opt = BallotDefinitions.Option(option.index, option.description, to_pos(new_x1,new_y1,new_x2,new_y2))
			# print("new option position",to_pos(new_x1,new_y1,new_x2,new_y2))
			converted_options.append(new_opt)

		new_contest = BallotDefinitions.Contest(contest.index, contest.contest_name, contest.description, converted_options, contest_pos)
		converted_contests.append(new_contest)
		# TODO: Determine where contest image is saved.
		new_contest.file = directory+"/cont"+str(cc)+".png"
		new_contest.image = contest_img.convert("RGBA")
		print(f"{new_contest.index}:  {new_contest.bounding_rect}")
		print(new_contest.image)
		for option in new_contest.options:
			print(f"{option.index}:  {option.bounding_rect}")
			# Future code expects that pixel positions  be integers, or image manipulations will fail.
			assert isinstance(option.bounding_rect.lower_right.y, int) and isinstance(option.bounding_rect.upper_left.y, int)
			assert isinstance(option.bounding_rect.lower_right.x, int) and isinstance(option.bounding_rect.upper_left.x, int)
			# Require that bounding rectangle be un-inverted. The upper left corner
			# must be strictly less than the bottom right corner.
			assert option.bounding_rect.lower_right.y > option.bounding_rect.upper_left.y
			assert option.bounding_rect.lower_right.x > option.bounding_rect.upper_left.x
			# Assert that option is contained entirely within a contest.
			assert new_contest.bounding_rect.upper_left.x < option.bounding_rect.upper_left.x
			assert new_contest.bounding_rect.upper_left.y < option.bounding_rect.upper_left.y
			assert new_contest.bounding_rect.lower_right.x > option.bounding_rect.lower_right.x
			assert new_contest.bounding_rect.lower_right.x > option.bounding_rect.lower_right.x
		print("\n")
		cc+=1
		# words = input("contest successfully created,\npress enter")
	ret_val = BallotDefinitions.Ballot(converted_contests, ballot.ballot_file)

	# Assert postconditions that all positions are now absolute, and that each contest has an image.
	for contest in ret_val.contests:
		assert isinstance(contest.bounding_rect, CNNScan.Ballot.Positions.PixelPosition)
		assert contest.image is not None
		for option in contest.options:
			assert isinstance(option.bounding_rect, CNNScan.Ballot.Positions.PixelPosition)
	return ret_val

# TODO: Convert a relative-positioned contest to an 
def rasterize_contest(Contest : BallotDefinitions.Contest, ballot_image : numpy.ndarray) -> typing.Tuple[numpy.ndarray, CNNScan.Ballot.Positions.PixelPosition]:
	return numpy.ndarray((0,0)), CNNScan.Ballot.Positions.PixelPosition()

