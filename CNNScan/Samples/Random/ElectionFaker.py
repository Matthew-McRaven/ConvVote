import random

import numpy as np
import numpy.random
from PIL import Image
from faker import Faker

from CNNScan.Ballot import BallotDefinitions, MarkedBallots, Positions
import CNNScan.Mark.Marks
# Create a single, fixed fake race with 4 candidates.
def create_fake_contest(pagenum, contest_index=0, min_candidate=1, max_candidates=8, min_xy_per_candidate=(18,16), max_xy_per_candidate=(64,16)):
	min_x, min_y = min_xy_per_candidate
	max_x, max_y = max_xy_per_candidate
	candidate_number = random.randint(min_candidate, max_candidates)
	fake = Faker()
	name = fake.catch_phrase()
	description = fake.text()
	options = []
	locations = []
	x_size = random.randint(min_x, max_x)
	y_rolling = 0
	for i in range(candidate_number):
		y_size = random.randint(min_y, max_y)
		bound = Positions.to_pixel_pos(0, y_rolling, x_size, y_rolling+y_size, pagenum)
		options.append(BallotDefinitions.Option(i, fake.name(), bounding_rect=(bound)))
		locations.append(bound)
		y_rolling += y_size
	print(f"{candidate_number} candidates, with a ballot that is {x_size}x{y_rolling}")
	contest = BallotDefinitions.Contest(contest_index, name=name, description=description,
		options=options, bounding_rect=Positions.to_pixel_pos(0,0, x_size, y_rolling, pagenum))
	return contest

def create_fake_ballot(factory, min_contests=3, max_contests=3)->BallotDefinitions.Ballot:
	contests = random.randint(min_contests, max_contests)
	contests_list = []
	for i in range(0, contests):
		current = create_fake_contest(i,contest_index=i)
		contests_list.append(current)

	ballot = factory.Ballot(contests_list)
	return ballot

# Create random noise with a dimensions matching that of the ballot.
def create_fake_contest_image(contest):
	# Pictures are stored in column major order, but numpy arrays are stored in row major order.
	# Must transpose for both kinds of images to compute correctly.
	# See:
	# 	https://stackoverflow.com/questions/19016144/conversion-between-pillow-image-object-and-numpy-array-changes-dimension
	# Additionally, PNG's contain data in [0, 255], so we must create an int ditribution to approximate this.
	shape = (contest.bounding_rect.lower_right.y, contest.bounding_rect.lower_right.x)
	r_data = numpy.random.randint(0,255, shape)   # Test data
	alpha = numpy.ndarray(shape)
	alpha.fill(255)
	r_data = numpy.stack((r_data, r_data, r_data, alpha), axis=2)
	return Image.fromarray(r_data, mode='RGBA')
	#return r_data