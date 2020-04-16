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
	raise NotImplementedError("This no longer does what you think it does")
	# Contest's image must already be streamed from disk.
	assert contest.image is not None
	# Duplicate template image, begin markup.
	ballot_image = contest.image.copy()
	# Determine probability of selecting no, one, or multiple options per contest
	return marked

def create_marked_ballot(ballot, mark_database):
	assert ballot.pages is not None
	contests = []
	marked_pages = [page.copy() for page in ballot.pages]
	for contest in ballot.contests:
		mark = mark_database.get_random_mark()
		count = np.random.choice([1], p=[1])
		# Generate random selections on ballot. Use set to avoid duplicates.
		selected = set()
		for i in range(count):
			selected.add(random.randint(0, len(contest.options) - 1))

		# MarkedContests needs selected indicies to be a list, not a set.
		marked = CNNScan.Ballot.MarkedBallots.MarkedContest(contest.index, None, list(selected))

		# For all the options that were selected for this contest, mark the contest.
		CNNScan.Mark.apply_marks(contest, marked, mark, marked_pages[contest.bounding_rect.page])

		contests.append(marked)
	return CNNScan.Ballot.MarkedBallots.MarkedBallot(ballot, contests, marked_pages)

# If lazy_nondeterminism is true, then ballots will only be loaded as they are referenced (a good option for files)
# If it is false, then all ballots will be loaded into memory before __init__ returns, 
# which is a good choice for generating ballots deterministically
class SingleBallotDataSet(Dataset):
	def __init__(self, ballot_count,  transforms, lazy_nondeterminism=True):
		self.ballot_count = ballot_count
		self.marked_ballots = [None] * ballot_count
		self.transforms = transforms
		self.lazy = lazy_nondeterminism

		# Force all ballots to be loaded immediately.
		if not lazy_nondeterminism:
			for i in range(self.ballot_count):
				self.at(i)

	# Querry for a ballot definition given only a ballot id
	def ballot_definition_from_id(self, ballot_def_id):
		raise NotImplementedError("")
	# Return all ballot definitions contained within this class as a list.
	def all_ballot_definitions(self):
		raise NotImplementedError("")

	# Given the index of a Ballot template, return the associated ballot template object
	def ballot_definition(self, index):
		raise NotImplementedError

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
			num_candidates = len(self.ballot_definition(index).contests[i].options)
			# TODO: Create additional encodings (such a multi-class classification or ranked-choice) that may be choosen from here.
			labels.append(torch.tensor(CNNScan.utils.labels_to_vec(contest.actual_vote_index, num_candidates), dtype=torch.float32))
			images.append(contest.tensor)
		return (index, labels, images)

	def __len__(self):
		return self.ballot_count

	# Determine the directory where a ballot is stored.
	def ballot_dir(self, directory, index):
		return directory + "/ballot%s/" % index

	# Determine the name for a contest's image
	def ballot_name(self, index):
		return f"ballot{index}.p"

	# Save all of the ballots/contests in the dataset to a file in a pre-determined fashion.
	def save_to_directory(self, output_directory, print_images=True):
		if not os.path.exists(output_directory):
			os.makedirs(output_directory)
		# TODO: use generator expression to create ballots in batches, to reduce memory pressure on host machine
		with open(output_directory+"/ballot-definition.p", "wb") as file:
			assert len(self.all_ballot_definitions()) == 1
			pickle.dump(self.all_ballot_definitions()[0], file)
		for i in range(len(self.marked_ballots)):
			marked_ballot = self.at(i)

			print(marked_ballot)

			# Require that ballot directories
			ballot_dir = self.ballot_dir(output_directory, i)
			if not os.path.exists(ballot_dir):
				os.mkdir(ballot_dir)

			# Serialize all contests to PNG's so they may be inspected.
			# Null out reference to image, so it does not get pickled.
			for j, page in enumerate(marked_ballot.pages):
				page.save(ballot_dir+f"b{j}.png")
			marked_ballot.pages = None

			# Remove contest image data; it can be cropped back from the marked pages
			for j, contest in enumerate(marked_ballot.marked_contest):
				contest.clear_data()
				
			# Serialize ballot (without images!) to object in directory structure
			with open(ballot_dir+self.ballot_name(i), "wb") as file:
				pickle.dump(marked_ballot, file)

class DirectoryDataSet(SingleBallotDataSet):
	def __init__(self, directory, transforms, lazy_nondeterminism=True):
		self.directory = directory
		self._ballot_definition = self.load_ballot_definition()
		super(DirectoryDataSet, self).__init__(self.count_ballots(), transforms, lazy_nondeterminism=lazy_nondeterminism)

	# Querry for a ballot definition given only a ballot id
	def ballot_definition_from_id(self, ballot_def_id):
		if self._ballot_definition.ballot_index == ballot_def_id:
			return self._ballot_definition
		else:
			raise KeyError(f"Could not find a ballot with id {ballot_def_id}.")
	# Return all ballot definitions contained within this class as a list.
	def all_ballot_definitions(self):
		return [self._ballot_definition]

	# Given the index of a Ballot template, return the associated ballot template object
	def ballot_definition(self, index):
		return self._ballot_definition

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

			# Ballot pages are not saved as part pickle'd format, so that they may be inspected by humans.
			marked.pages = []
			print(self._ballot_definition.pages)
			for i, _ in enumerate(self._ballot_definition.pages):
				marked.pages.append(Image.open(directory+f"b{i}.png"))

			# Crop pages of marked ballots to contain only individual contests.

			CNNScan.Raster.Raster.crop_contests(self._ballot_definition, marked)

			# Make sure that unpickle'ing gave us a usable ballot.
			assert isinstance(marked, CNNScan.Ballot.MarkedBallots.MarkedBallot)
			
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

class GeneratingDataSet(SingleBallotDataSet):
	def __init__(self, ballot_def, markdb, ballot_count, transforms, lazy_nondeterminism=True):
		self.markdb = markdb
		self._ballot_definition = ballot_def
		super(GeneratingDataSet, self).__init__(ballot_count, transforms, lazy_nondeterminism=lazy_nondeterminism)

	# Querry for a ballot definition given only a ballot id
	def ballot_definition_from_id(self, ballot_def_id):
		if self._ballot_definition.ballot_index == ballot_def_id:
			return self._ballot_definition
		else:
			raise KeyError(f"Could not find a ballot with id {ballot_def_id}.")
	# Return all ballot definitions contained within this class as a list.
	def all_ballot_definitions(self):
		return [self._ballot_definition]

	# Given the index of a Ballot template, return the associated ballot template object
	def ballot_definition(self, index):
		return self._ballot_definition

	def load_ballot(self, index):
		# We shouldn't be asked to create a new ballot when one already exists.
		assert self.marked_ballots[index] is None
		ret_val = create_marked_ballot(self._ballot_definition, self.markdb)
		CNNScan.Raster.Raster.crop_contests(self._ballot_definition, ret_val)
		return ret_val

class MultiGeneratingDataSet(Dataset):
	def __init__(self, ballot_definition_list, markdb, count_list, transforms, lazy_nondeterminism=True):
		self.markdb = markdb
		assert len(ballot_definition_list) == len(count_list)
		self.datasets = []
		self.ballot_count = 0
		self._ballot_definitions = ballot_definition_list
		for definition, count in zip(ballot_definition_list, count_list):
			self.ballot_count += count
			self.datasets.append(GeneratingDataSet(definition, self.markdb, count, transforms, lazy_nondeterminism=lazy_nondeterminism))

	# Querry for a ballot definition given only a ballot id
	def ballot_definition_from_id(self, ballot_def_id):
		for ballot in self._ballot_definition:
			if ballot.ballot_index == ballot_def_id:
				return ballot
		raise KeyError(f"Could not find a ballot with id {ballot_def_id}.")

	# Return all ballot definitions contained within this class as a list.
	def all_ballot_definitions(self):
		return self._ballot_definition

	# Given the index of a Ballot template, return the associated ballot template object
	def ballot_definition(self, index):
		count = 0
		for dataset in self.datasets:
			if index - count < len(dataset):
				return dataset.ballot_definition(index-count)
			else:
				count += len(dataset)
	
	# Fetch the item at index, which may not yet be loaded into memory.
	def at(self, index):
		count = 0
		for dataset in self.datasets:
			if index - count < len(dataset):
				return dataset.at(index-count)
			else:
				count += len(dataset)

	# Return tensors usable by DataLoader interface.
	def __getitem__(self, index):
		val = self.at(index)
		labels, images =[],[]
		for i, contest in enumerate(val.marked_contest):
			num_candidates = len(self.ballot_definition(index).contests[i].options)
			# TODO: Create additional encodings (such a multi-class classification or ranked-choice) that may be choosen from here.
			labels.append(torch.tensor(CNNScan.utils.labels_to_vec(contest.actual_vote_index, num_candidates), dtype=torch.float32))
			images.append(contest.tensor)
		return (index, labels, images)

	def __len__(self):
		return self.ballot_count