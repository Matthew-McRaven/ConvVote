from typing import List
import importlib.resources
import random

import torch
import numpy as np
import numpy.random
import pylab as plt

from PIL import Image

from CNNScan.Ballot import MarkedBallots, BallotDefinitions
import CNNScan.Mark

def load_template_image(package, contest:BallotDefinitions.Contest) -> np.ndarray:
	a = importlib.resources.open_binary(package, contest.contest_file)
	image = Image.open(a)
	data = np.array(image, dtype='uint8')
	real_data = np.ndarray((data.shape[1], data.shape[0]))
	# Mask out alpha channel of PNG, since it provides no useful information on the images we have.
	# Channels are ordered RGBA.
	weights = np.asarray([1,1,1,0])
	rval = np.average(data, axis=-1, weights=weights,returned=True)[0]
	real_data = rval

	return real_data

# Create a fake ballot image, and select a random candiate to win.
# Black out all the pixels corresponding to the location on the ballot representing the candidate.
def create_fake_marked_contest(package, mark, contest:BallotDefinitions.Contest):

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
	CNNScan.Mark.apply_marks(marked, mark)
	return marked

def make_sample_ballots(module, ballot:BallotDefinitions.Ballot, count=100) -> List[MarkedBallots.MarkedBallot]:
	mark_db = CNNScan.Mark.MarkDatabase()
	mark_db.insert_mark(CNNScan.Mark.BoxMark())
	ballots =  module.create_marked_ballots(ballot, mark_db, count)
	return ballots