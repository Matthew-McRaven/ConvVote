from typing import List
import importlib.resources
import random

import torch
import numpy as np
import numpy.random
import pylab as plt

from torch.utils.data import Dataset, DataLoader

import CNNScan.Ballot.MarkedBallots

# Create a fake ballot image, and select a random candiate to win.
# Black out all the pixels corresponding to the location on the ballot representing the candidate.
def create_marked_contest(mark, contest:CNNScan.Ballot.BallotDefinitions.Contest):
	# Contest's image must already be streamed from disk.
	assert contest.image is not None
	# Duplicate template image, begin markup.
	ballot_image = contest.image.copy()
	# Determine probability of selecting no, one, or multiple options per contest
	count = np.random.choice([1], p=[1])
	# Generate random selections on ballot. Use set to avoid duplicates.
	selected = set()
	for i in range(count):
		selected.add(random.randint(0, len(contest.options) - 1))

	# MarkedContests needs selected indicies to be a list, not a set.
	marked = CNNScan.Ballot.MarkedBallots.MarkedContest(contest, ballot_image, list(selected))

	# For all the options that were selected for this contest, mark the contest.
	CNNScan.Mark.apply_marks(marked, mark)
	return marked

def create_marked_ballot(ballot, mark_database):
	contests = []
	for contest in ballot.contests:
		mark = mark_database.get_random_mark()
		latest = create_marked_contest(mark, contest)
		contests.append(latest)
	return CNNScan.Ballot.MarkedBallots.MarkedBallot(ballot, contests)



class ImageDataSet(Dataset):
	def __init__(self, ballot_def, markdb, ballot_count, transform):
		# TODO: Add configuration parameters to control markings.
		self.ballot = ballot_def
		self.markdb = markdb
		self.ballot_count = ballot_count
		# Initialize list of ballot objects to nothing.
		# Lazily load/generate referenced ballot on first use
		self.marked_ballots = ballot_count * [None]
		# torchvision.transfom to convernt 4 channel RGBA image to a tensor.
		self.transform = transform
		
	# Needed to refrence ballot object rather than getting tensor-like data used by DataLoader
	def at(self, index):
		if self.marked_ballots[index] is None:
			self.marked_ballots[index] = create_marked_ballot(self.ballot, self.markdb)
			for contest in self.marked_ballots[index].marked_contest:
				contest.tensor = self.transform(contest.image)
		return self.marked_ballots[index]

	# Return tensors usable by DataLoader interface.
	def __getitem__(self, index):
		val = self.at(index)
		labels, images =[],[]
		for contest in val.marked_contest:
			num_candidates = len(contest.contest.options)
			# TODO: Create additional encodings (such a multi-class classification or ranked-choice) that may be choosen from here.
			labels.append(torch.tensor(CNNScan.utils.labels_to_vec(contest.actual_vote_index, num_candidates), dtype=torch.float32))
			images.append(contest.tensor)
		return (index, labels, images)

	def __len__(self):
		return self.ballot_count