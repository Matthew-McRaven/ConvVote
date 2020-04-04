
import Election as Election
import numpy.random
import random

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
	for x in range(location[0], location[0]+location[2]):
		for y in range(location[1], location[1]+location[3]):
			marked.image[y][x]=0
	return marked


# Create multiple fake ballots using create_fake_marked_ballot()
def create_fake_marked_ballots(contest, contest_phys_data, count):
	return [create_fake_marked_ballot(contest, contest_phys_data) for i in range(count)]

# Create a single, fixed fake race with 4 candidates.
def create_fake_contest():
	name = "Governor"
	description = "This race represents the race for govenor. Instructions here blah blah blah."
	options = []
	options.append(Election.OptionDefinition(0, "Candidate A"))
	options.append(Election.OptionDefinition(1, "Candidate B"))
	options.append(Election.OptionDefinition(2, "Candidate C"))
	options.append(Election.OptionDefinition(3, "Candidate D"))
	contest = Election.ContestDefinition(0, name, description, options)
	
	locations = [(0, 0 ,64, 16), (0, 16 ,64, 16), (0, 32 ,64, 16), (0, 48 ,64, 16)]
	contest_phys = Election.ContestLocation(0, (0,0, 64, 64), locations)
	return (contest, contest_phys)