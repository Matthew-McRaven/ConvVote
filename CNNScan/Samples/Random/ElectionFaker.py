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
	x_size = random.randint(min_x, max_x)
	y_rolling = 0
	sizes = [random.randint(min_y, max_y) for y in range(candidate_number)]
	max_y_size = np.sum(sizes)
	for i in range(candidate_number):
		y_size = sizes[i]
		rel_bound = Positions.to_percent_pos(0, y_rolling/max_y_size, 1, (y_rolling+y_size)/max_y_size, pagenum)
		abs_bound = Positions.to_pixel_pos(0, y_rolling, max_x, y_rolling+y_size, pagenum)
		new_option = BallotDefinitions.Option(i, fake.name(), rel_bounding_rect=(rel_bound))
		new_option.abs_bounding_rect = abs_bound
		options.append(new_option)
		y_rolling += y_size
	print(f"{candidate_number} candidates, with a ballot that is {x_size}x{y_rolling}")

	abs_bound = Positions.to_pixel_pos(0,0, x_size, y_rolling, pagenum)
	contest = BallotDefinitions.Contest(contest_index, name=name, description=description,
		options=options, rel_bounding_rect=Positions.to_percent_pos(0,0,1,1,pagenum))
	contest.abs_bounding_rect = abs_bound
	return contest

def create_fake_ballot(factory, min_contests=2, max_contests=8)->BallotDefinitions.Ballot:
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
	shape = (contest.abs_bounding_rect.lower_right.y, contest.abs_bounding_rect.lower_right.x)
	r_data = numpy.random.randint(0,255, shape)   # Test data
	alpha = numpy.ndarray(shape)
	alpha.fill(255)
	r_data = numpy.stack((r_data, r_data, r_data, alpha), axis=2)
	return Image.fromarray(r_data, mode='RGBA')
	#return r_data