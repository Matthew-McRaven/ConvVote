
import Election as Election
import numpy.random
import random
from faker import Faker

# TODO: Not implemented
def create_election()->Election.Ballot:
	election = Election.Ballot()
	return election

# Create random noise with a dimensions matching that of the ballot.
def create_fake_ballot_image(contest):
	r_data = numpy.random.random((contest.bound_rect[2], contest.bound_rect[3]))   # Test data
	return r_data

# Create a fake ballot image, and select a random candiate to win.
# Black out all the pixels corresponding to the location on the ballot representing the candidate.
def create_fake_marked_ballot(contest):
	ballot_image = create_fake_ballot_image(contest)
	marked = Election.MarkedContest(contest, ballot_image, random.randint(0, len(contest.options)))
	if marked.actual_vote_index == len(contest.options):
		marked.actual_vote_index = None
		return marked
	which = marked.actual_vote_index
	location = contest.options[which].bound_rect
	for x in range(location[0], location[2]):
		for y in range(location[1], location[3]):
			marked.image[x][y]=0
	return marked


# Create multiple fake ballots using create_fake_marked_ballot()
def create_fake_marked_ballots(contest, count):
	return [create_fake_marked_ballot(contest) for i in range(count)]

# Create a single, fixed fake race with 4 candidates.
def create_fake_contest(contest_index=0, min_candidate=1, max_candidates=15, min_xy_per_candidate=(18,4), max_xy_per_candidate=(64,16)):
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
		bound = (0, y_rolling, x_size, y_rolling+y_size)
		options.append(Election.Option(i, fake.name(), bounding_rect=(bound)))
		locations.append(bound)
		y_rolling += y_size
	print(f"{candidate_number} candidates, with a ballot that is {x_size}x{y_rolling}")
	contest = Election.Contest(contest_index, name=name, description=description, options=options, bounding_rect=(0,0, x_size, y_rolling))
	return contest