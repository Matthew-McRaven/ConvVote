from typing import List
import importlib.resources
import random
import pickle
import os.path
import re

import numpy as np
import numpy.random
import pylab as plt
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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

# If lazy_nondeterminism is true, then ballots will only be loaded as they are referenced (a good option for files)
# If it is false, then all ballots will be loaded into memory before __init__ returns, 
# which is a good choice for generating ballots deterministically
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

	# Subclasses must override this in order to support lazy-loading of ballots
	def load_ballot(self, index):
		raise NotImplementedError

	# Fetch the item at index, which may not yet be loaded into memory.
	def at(self, index):
		if self.marked_ballots[index] is None:
			self.marked_ballots[index] = self.load_ballot(index)
			# Tensors were purged on save, so they must be re-constructed.
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
	
	# Save all of the ballots/contests in the dataset to a file in a pre-determined fashion.
	def save_to_directory(self, output_directory, print_images=True):
		if not os.path.exists(output_directory):
			os.mkdir(output_directory)
		# TODO: use generator expression to create ballots in batches, to reduce memory pressure on host machine
		with open(output_directory+"/ballot-definition.p", "wb") as file:
			pickle.dump(self.ballot_definition, file)
		for i in range(len(self.marked_ballots)):
			marked_ballot = self.at(i)
			print(marked_ballot)
			# Serialize all contests to PNG's so they may be inspected.
			# Null out reference to image, so it does not get pickled.
			ballot_dir = self.ballot_dir(output_directory, i)
			if not os.path.exists(ballot_dir):
				os.mkdir(ballot_dir)
				for j, contest in enumerate(marked_ballot.marked_contest):
					print(ballot_dir+f"c{j}.png")
					contest.image.save(ballot_dir+f"c{j}.png")
					contest.clear_data()
				
			# Serialize ballot (without images!) to object in directory structure
			with open(ballot_dir+self.ballot_name(i), "wb") as file:
				pickle.dump(marked_ballot, file)

	# Determine the directory where a ballot is stored.
	def ballot_dir(self, directory, index):
		return directory + "/ballot%s/" % index

	# Determine the name for a contest's image
	def ballot_name(self, index):
		return f"ballot{index}.p"

class DirectoryDataSet(BallotDataSet):
	def __init__(self, directory, transforms, lazy_nondeterminism=True):
		self.directory = directory
		super(DirectoryDataSet, self).__init__(self.load_ballot_definition(), self.count_ballots(), transforms, lazy_nondeterminism=lazy_nondeterminism)

	# Load a marked ballot with a particular index.
	def load_ballot(self, index):
		# We shouldn't be asked to create a new ballot when one already exists
		assert self.marked_ballots[index] is None
		directory = self.ballot_dir(self.directory, index)
		assert os.path.exists(directory)
		fp = self.ballot_name(index)
		assert os.path.isfile(directory+fp)
		with open(directory+fp, "rb") as file:
			marked = pickle.load(file)
			assert isinstance(marked, CNNScan.Ballot.MarkedBallots.MarkedBallot)
			# TODO: Save entire pages of ballot, crop from entire page of ballot.
			# Must re-load all the pictures for each contest.
			for i, contest in enumerate(marked.marked_contest):
				contest.image = Image.open(directory+f"c{i}.png")
			#print(marked)
			return marked

	# Load and return the object containing the ballot definition.
	def load_ballot_definition(self):
		if os.path.exists(self.directory+"/ballot-definition.p"):
			with open(self.directory+"/ballot-definition.p", "rb") as file:
				template = pickle.load(file)
				assert isinstance(template, CNNScan.Ballot.BallotDefinitions.Ballot)
				#print(template)
				return template
		else:
			raise ValueError(f"No ballot definition found at {self.directory}")
		raise NotImplementedError("Can't yet read from directories")

	# Figure out the number of marked ballot files in the current directory.
	def count_ballots(self):
		dirs = os.listdir(path=self.directory)
		count = 0
		program = re.compile("ballot[0-9]+")
		for x in dirs:
			if program.match(x):
				count += 1
		#print(count)
		return count


class GeneratingDataSet(BallotDataSet):
	def __init__(self, ballot_def, markdb, ballot_count, transforms, lazy_nondeterminism=True):
		self.markdb = markdb
		super(GeneratingDataSet, self).__init__(ballot_def, ballot_count, transforms, lazy_nondeterminism=lazy_nondeterminism)

	def load_ballot(self, index):
		# We shouldn't be asked to create a new ballot when one already exists.
		assert self.marked_ballots[index] is None
		return create_marked_ballot(self.ballot_definition, self.markdb)
