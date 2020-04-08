
import numpy.random
import random
from faker import Faker

import numpy as np
import numpy.random

from CNNScan.Ballot import BallotDefinitions, MarkedBallots, Positions

# Create a single, fixed fake race with 4 candidates.
def create_fake_contest(contest_index=0, min_candidate=1, max_candidates=1, min_xy_per_candidate=(18,8), max_xy_per_candidate=(64,16)):
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
		bound = Positions.to_pixel_pos(0, y_rolling, x_size, y_rolling+y_size)
		options.append(BallotDefinitions.Option(i, fake.name(), bounding_rect=(bound)))
		locations.append(bound)
		y_rolling += y_size
	print(f"{candidate_number} candidates, with a ballot that is {x_size}x{y_rolling}")
	contest = BallotDefinitions.Contest(contest_index, name=name, description=description,
		options=options, bounding_rect=Positions.to_pixel_pos(0,0, x_size, y_rolling))
	return contest

def create_fake_ballot(min_contests=3, max_contests=3)->BallotDefinitions.Ballot:
	contests = random.randint(min_contests, max_contests)
	contests_list = []
	for i in range(0, contests):
		current = create_fake_contest(contest_index=i)
		contests_list.append(current)

	ballot = BallotDefinitions.Ballot(contests_list)
	return ballot

# Create random noise with a dimensions matching that of the ballot.
def create_fake_contest_image(contest):
	r_data = numpy.random.random((contest.bounding_rect.lower_right.x, contest.bounding_rect.lower_right.y))   # Test data
	return r_data

# Create a fake ballot image, and select a random candiate to win.
# Black out all the pixels corresponding to the location on the ballot representing the candidate.
def create_fake_marked_contest(contest):
	ballot_image = create_fake_contest_image(contest)
	# Determine probability of selecting no, one, or multiple options per contest
	count = np.random.choice([0,1,2,3], p=[.1,.6,.2,.1])
	# Generate random selections on ballot. Use set to avoid duplicates.
	selected = set()
	for i in range(count):
		selected.add(random.randint(0, len(contest.options) - 1))

	# MarkedContests needs selected indicies to be a list, not a set.
	marked = MarkedBallots.MarkedContest(contest, ballot_image, list(selected))

	# For all the options that were selected for this contest, mark the contest.
	for which in marked.actual_vote_index:
		location = contest.options[which].bounding_rect
		for x in range(location.upper_left.x, location.lower_right.x):
			for y in range(location.upper_left.y, location.lower_right.y):
				marked.image[x][y]=0
	# Pictures are stored in column major order, but numpy arrays are stored in row major order.
	# Must transpose for both kinds of images to compute correctly.
	# See:
	# 	https://stackoverflow.com/questions/19016144/conversion-between-pillow-image-object-and-numpy-array-changes-dimension
	marked.image = np.transpose(marked.image)
	return marked


# Create multiple fake ballots using create_fake_marked_ballot()
def create_fake_marked_contests(contest, count):
	return [create_fake_marked_contest(contest) for i in range(count)]

# Create a single fake ballot
def create_fake_marked_ballot(ballot):
	marked = []
	for index, contest in enumerate(ballot.contests):
		marked.append(create_fake_marked_contest(contest))
	return MarkedBallots.MarkedBallot(ballot, marked)

def create_fake_marked_ballots(ballot, count):
	return [create_fake_marked_ballot(ballot) for i in range(count)]