
import Election as Election
import numpy.random
import random
from faker import Faker

# TODO: Not implemented
def create_election()->Election.BallotDefinition:
	election = Election.BallotDefinition()
	return election

# Create random noise with a dimensions matching that of the ballot.
def create_fake_ballot_image(contest, contest_phys_data):
	r_data = numpy.random.random((contest_phys_data.bound_rect[2], contest_phys_data.bound_rect[3]))   # Test data
	return r_data

# Create a fake ballot image, and select a random candiate to win.
# Black out all the pixels corresponding to the location on the ballot representing the candidate.
def create_fake_marked_ballot(contest, contest_phys_data):
	ballot_image = create_fake_ballot_image(contest, contest_phys_data)
	marked = Election.MarkedContest(contest, ballot_image, random.randint(0, len(contest.options)-1))
	which = marked.actual_vote_index
	location = contest_phys_data.options[which]
	for x in range(location[0], location[2]):
		for y in range(location[1], location[3]):
			marked.image[x][y]=0
	return marked


# Create multiple fake ballots using create_fake_marked_ballot()
def create_fake_marked_ballots(contest, contest_phys_data, count):
	return [create_fake_marked_ballot(contest, contest_phys_data) for i in range(count)]

# Create a single, fixed fake race with 4 candidates.
def create_fake_contest(contest_index=0, min_candidate=1, max_candidates=15, min_xy_per_candidate=(18,4), max_xy_per_candidate=(64,16)):
	min_x, min_y = min_xy_per_candidate
	max_x, max_y = max_xy_per_candidate
	candidate_number = random.randint(min_candidate, max_candidates)
	x_size = random.randint(min_x, max_x)
	y_size = random.randint(min_y, max_y)
	fake = Faker()
	name = fake.catch_phrase()
	description = fake.text()
	options = []
	locations = []
	y_offset = 0
	for i in range(candidate_number):
		options.append(Election.OptionDefinition(i, fake.name()))
		locations.append((0, y_offset, x_size, y_offset+y_size))
		y_offset += y_size
	print(f"{candidate_number} candidates, each with a box that is {x_size}x{y_size}")
	contest = Election.ContestDefinition(contest_index, name, description, options)
	contest_phys = Election.ContestLocation(contest_index, (0,0, x_size, y_size * candidate_number), locations)
	return (contest, contest_phys)