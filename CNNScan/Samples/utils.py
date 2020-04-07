from typing import List
import importlib.resources
import random

import torch
import numpy as np
import numpy.random
import pylab as plt

from PIL import Image

from CNNScan.Ballot import MarkedBallots, BallotDefinitions

def load_template_image(package, contest:BallotDefinitions.Contest) -> np.ndarray:
	a = importlib.resources.open_binary(package, contest.contest_file)
	image = Image.open(a)
	data = np.array(image, dtype='uint8')
	real_data = np.ndarray((data.shape[1], data.shape[0]))
	#print("img")
	weights = np.asarray([1,1,1,0])
	rval = np.average(data, axis=-1, weights=weights,returned=True)[0]
	#print(rval.shape)
	real_data = np.transpose(rval)
	"""	print(rval)
	for x in range(data.shape[1]):
		for y in range (data.shape[0]):
			real_data[x][y] = 0
	print("Done")"""
	return real_data

# Create a fake ballot image, and select a random candiate to win.
# Black out all the pixels corresponding to the location on the ballot representing the candidate.
def create_fake_marked_contest(package, contest:BallotDefinitions.Contest):
	if contest.image is None:
		contest.image = load_template_image(package, contest)
	ballot_image = np.copy(contest.image)
	# Determine probability of selecting no, one, or multiple options per contest
	count = np.random.choice([1], p=[1])
	# Generate random selections on ballot. Use set to avoid duplicates.
	selected = set()
	for i in range(count):
		selected.add(random.randint(0, len(contest.options) - 1))

	# MarkedContests needs selected indicies to be a list, not a set.
	marked = MarkedBallots.MarkedContest(contest, ballot_image, list(selected))

	# For all the options that were selected for this contest, mark the contest.
	for which in marked.actual_vote_index:
		location = contest.options[which].bounding_rect
		for x in range(location[0], location[2]):
			for y in range(location[1], location[3]):
				marked.image[x][y]=0
				#marked.image[y][x][1]=0
				#marked.image[y][x][2]=0
	return marked

def make_sample_ballots(package, ballot:BallotDefinitions.Ballot, number=100) -> List[MarkedBallots.MarkedBallot]:
	ballots = []
	for i in range(number):
		contests = []
		for contest in ballot.contests:
			latest = create_fake_marked_contest(package, contest)
			contests.append(latest)
			#plt.title(f'{i}-th copy of Contest {contest.index}')
			#plt.xlabel(f'Recorded as a vote for {latest.actual_vote_index}')
			#plt.imshow(latest.image, cmap='gray', interpolation='nearest')
			#plt.show()
		new_ballot = MarkedBallots.MarkedBallot(ballot, contests)
		ballots.append(new_ballot)
	return ballots