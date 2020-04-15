from typing import List
import importlib.resources
import random
import pickle

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
	marked = CNNScan.Ballot.MarkedBallots.MarkedContest(contest.index, ballot_image, list(selected))

	# For all the options that were selected for this contest, mark the contest.
	CNNScan.Mark.apply_marks(contest, marked, mark)
	return marked

def create_marked_ballot(ballot, mark_database):
	contests = []
	for contest in ballot.contests:
		mark = mark_database.get_random_mark()
		latest = create_marked_contest(mark, contest)
		contests.append(latest)
	return CNNScan.Ballot.MarkedBallots.MarkedBallot(ballot, contests)

class BallotDataSet(Dataset):
	def __init__(self, ballot_definition, ballot_count,  transforms, lazy_nondeterminism=True):
		self.ballot_definition = ballot_definition
		self.ballot_count = ballot_count
		self.marked_ballots = [None] * ballot_count
		self.transforms = transforms
		self.lazy = lazy_nondeterminism

		# Force all ballots to be loaded immediately.
		if not lazy_nondeterminism:
			for i in range(self.ballot_count):
				self.at(i)

	def load_ballot(self, index):
		raise NotImplementedError

	def at(self, index):
		if self.marked_ballots[index] is None:
			self.marked_ballots[index] = self.load_ballot(index)
			for contest in self.marked_ballots[index].marked_contest:
				contest.tensor = self.transforms(contest.image)
		return self.marked_ballots[index]

	# Return tensors usable by DataLoader interface.
	def __getitem__(self, index):
		val = self.at(index)
		labels, images =[],[]
		for i, contest in enumerate(val.marked_contest):
			num_candidates = len(self.ballot_definition.contests[i].options)
			# TODO: Create additional encodings (such a multi-class classification or ranked-choice) that may be choosen from here.
			labels.append(torch.tensor(CNNScan.utils.labels_to_vec(contest.actual_vote_index, num_candidates), dtype=torch.float32))
			images.append(contest.tensor)
		return (index, labels, images)

	def __len__(self):
		return self.ballot_count

class DirectoryDataSet(BallotDataSet):
	def __init__(self, directory, transforms, lazy_nondeterminism=True):
		self.directory = directory
		super(DirectoryDataSet, self).__init__(self.load_ballot_def(), self.count_ballots(), transforms, lazy_nondeterminism=lazy_nondeterminism)

	def load_ballot(self, index):
		# We shouldn't be asked to create a new ballot when one already exists
		assert self.marked_ballots[index] is None
		raise NotImplementedError("Can't yet read from directories")

	def load_ballot_definition(self):
		raise NotImplementedError("Can't yet read from directories")

	def count_ballots(self):
		raise NotImplementedError("Can't yet read from directories")


class GeneratingDataSet(BallotDataSet):
	def __init__(self, ballot_def, markdb, ballot_count, transforms, lazy_nondeterminism=True):
		self.markdb = markdb
		super(GeneratingDataSet, self).__init__(ballot_def, ballot_count, transforms, lazy_nondeterminism=lazy_nondeterminism)

	def load_ballot(self, index):
		# We shouldn't be asked to create a new ballot when one already exists.
		assert self.marked_ballots[index] is None
		return create_marked_ballot(self.ballot_definition, self.markdb)
