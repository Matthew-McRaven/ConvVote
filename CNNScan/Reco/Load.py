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
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import CNNScan.Ballot.MarkedBallots
def_trans =  torchvision.transforms.Compose([#torchvision.transforms.Lambda(lambda x: np.average(x, axis=-1, weights=[1,1,1,0],returned=True)[0]),
					                         torchvision.transforms.ToTensor(),
											 torchvision.transforms.Lambda(lambda x: x.float()),
											 torchvision.transforms.Normalize((1,),(127.5,))
											 #torchvision.transforms.Lambda(lambda x: (1.0 - (x / 127.5)).float())
											])
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
		CNNScan.Mark.apply_marks(contest, marked, mark, marked_pages[contest.abs_bounding_rect.page])

		contests.append(marked)
	return CNNScan.Ballot.MarkedBallots.MarkedBallot(ballot.ballot_index, contests, marked_pages)

# Describe a dataset containing multiple ballot definitions that is lazily loaded/generated.
class MixedBallotDataSet(Dataset):
	def __init__(self, ballot_factory, ballot_counts, transforms=def_trans, lazy_nondeterminism=True):
		self.ballot_factory = ballot_factory
		self.ballot_count = ballot_counts
		self.marked_ballots = [[None] * ballot_count for ballot_count in ballot_counts]
		self.transforms = transforms
		self.lazy = lazy_nondeterminism
		self.fixed_ballot = None

		# Determine
		self.ballot_offsets = []
		start = 0
		for count in ballot_counts:
			self.ballot_offsets.append(start)
			start += count

		# Force all ballots to be loaded immediately.
		if not lazy_nondeterminism:
			for ballot_type in range(len(self.ballot_factory.ballots)):
				self.freeze_ballot_definiton_index(ballot_type)
				for i in range(self.__len__()):
					self.at(i)
		
	# Make the dataset behave as if it only contained ballots for "which"
	# Len, [] operations will only factor in selected ballot.
	def freeze_ballot_definiton_index(self, which):
		assert which >= 0 and which<len(self.ballot_factory)
		self.fixed_ballot = which

	# Querry for a ballot definition given only a ballot id
	def ballot_definition_from_id(self, ballot_index):
		assert ballot_index >=0 and ballot_index < len(self.ballot_factory)
		return self.ballot_factory.ballots[ballot_index]

	# Return all ballot definitions contained within this class as a list.
	def all_ballot_definitions(self):
		return self.ballot_factory.ballots

	# Given the index of a Ballot template, return the associated ballot definition object
	def ballot_definition(self, index):
		ballot_index = self.at(index, self.fixed_ballot).ballot_index
		return self.ballot_definition_from_id(ballot_index)

	# Fetch the item at index, which may not yet be loaded into memory.
	def at(self, index, ballot_number=None):
		# If no "active" ballot was specified,
		# find which set of ballots the index is
		# contained in, and update index, ballot_number
		if ballot_number is None:
			ballot_number = 0
			best_offset = 0
			for offset in self.ballot_offsets:
				if offset > index:
					break
				ballot_number += 1
				best_offset = offset
			ballot_number -= 1
			index -= best_offset

		# Load ballot if it has not yet been paged in.
		if self.marked_ballots[ballot_number][index] is None:
			self.marked_ballots[ballot_number][index] = self.load_ballot(index, ballot_number)
			# Tensors were purged on save, so they must be re-constructed.
			for contest in self.marked_ballots[ballot_number][index].marked_contest:
				contest.tensor = self.transforms(contest.image)

		return self.marked_ballots[ballot_number][index]

	# Return the length of the active ballot, or return the total number of cached ballots.
	def __len__(self):
		if self.fixed_ballot is None:
			return np.sum(self.ballot_count)
		else:
			return self.ballot_count[self.fixed_ballot]

	# Return tensors usable by DataLoader interface.
	def __getitem__(self, index):
		val = self.at(index, self.fixed_ballot)
		labels, images =[],[]
		for i, contest in enumerate(val.marked_contest):
			num_candidates = len(self.ballot_definition(index).contests[i].options)
			# TODO: Create additional encodings (such a multi-class classification or ranked-choice) that may be choosen from here.
			#labels.append(torch.tensor(CNNScan.utils.labels_to_vec(contest.actual_vote_index, num_candidates), dtype=torch.float32))
			# TODO: Report more than one label per contest.
			labels.append(torch.tensor(contest.actual_vote_index[0], dtype=torch.long))
			images.append(contest.tensor)
		return (index, val.ballot_index, labels, images)

	# Determine where a marked ballot is stored.
	def ballot_dir(self, directory, index, ballot_number):
		return directory + "/def%s"%ballot_number+"/ballot%s/" % index

	# Determine the name for an individual marked ballot definition.
	def ballot_name(self, index):
		return f"ballot{index}.p"

	# Save all of the ballots/contests in the dataset to a file in a pre-determined fashion.
	def save_to_directory(self, output_directory, print_images=True):
		if not os.path.exists(output_directory):
			os.makedirs(output_directory)
		# TODO: use generator expression to create ballots in batches, to reduce memory pressure on host machine
		with open(output_directory+"/ballot-definition.p", "wb") as file:
			pickle.dump(self.ballot_factory, file)

		# Save all marked ballots for each ballot definition.
		for ballot_number in range(len(self.marked_ballots)):
			# Override the active ballot definition, so len(self) will work correctly.
			self.freeze_ballot_definiton_index(ballot_number)
			for i in range(len(self)):
				# Must explicitly pass in ballot number, or only the first ballot definition will be touched.
				marked_ballot = self.at(i, ballot_number)

				# Make ballot directories if they do not exist.
				ballot_dir = self.ballot_dir(output_directory, i, ballot_number)
				if not os.path.exists(ballot_dir):
					os.makedirs(ballot_dir)

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
	
# If lazy_nondeterminism is true, then ballots will only be loaded as they are referenced (a good option for files)
# If it is false, then all ballots will be loaded into memory before __init__ returns, 
# which is a good choice for generating ballots deterministically
class DirectoryDataSet(MixedBallotDataSet):
	def __init__(self, directory, transforms=def_trans, lazy_nondeterminism=True):
		self.directory = directory
		super(DirectoryDataSet, self).__init__(self.load_ballot_factory(), self.count_ballots(), transforms, lazy_nondeterminism=lazy_nondeterminism)

	# Load a marked ballot with a particular index.
	def load_ballot(self, index, ballot_number):
		# We shouldn't be asked to create a new ballot when one already exists
		assert self.marked_ballots[ballot_number][index] is None
		directory = self.ballot_dir(self.directory, index, ballot_number)
		assert os.path.exists(directory)
		fp = self.ballot_name(index)
		assert os.path.isfile(directory+fp)

		with open(directory+fp, "rb") as file:
			marked = pickle.load(file)
			# Make sure that unpickle'ing gave us a usable ballot.
			assert isinstance(marked, CNNScan.Ballot.MarkedBallots.MarkedBallot)

			# Determine which ballot definition is paired with this marked ballot.
			ballot_index = marked.ballot_index
			assert ballot_index >= 0 and ballot_index < len(self.ballot_factory)
			ballot_definition = self.ballot_factory.ballots[ballot_index]
			# Ballot pages are not saved as part pickle'd format, so that they may be inspected by humans.
			marked.pages = []
			for i, _ in enumerate(self.ballot_factory.ballots[marked.ballot_index].pages):
				marked.pages.append(Image.open(directory+f"b{i}.png"))

			# Crop pages of marked ballots to contain only individual contests.
			CNNScan.Raster.Raster.crop_contests(ballot_definition, marked)

			
			return marked

	# Load and return the object containing the ballot definition.
	def load_ballot_factory(self):
		if os.path.exists(self.directory+"/ballot-definition.p"):
			with open(self.directory+"/ballot-definition.p", "rb") as file:
				template = pickle.load(file)
				assert isinstance(template, CNNScan.Ballot.BallotDefinitions.BallotFactory)
				return template
		else:
			raise ValueError(f"No ballot factory definition found at {self.directory}")

	# Figure out the number of marked ballot files in the current directory.
	def count_ballots(self):

		outer = self.directory
		count_list = []
		for inner in os.listdir(path=outer):
			inner = outer+"/"+inner
			#print(inner)
			program = re.compile("def[0-9]+")
			if not os.path.isdir(inner):
				print(f"{inner} is not a dir.")
				continue

			ballot_dirs = os.listdir(path=inner)
			count = 0
			program = re.compile("ballot[0-9]+")
			for x in ballot_dirs:
				if program.match(x):
					count += 1
			count_list.append(count)
		print(count_list)
		return count_list


class GeneratingDataSet(MixedBallotDataSet):
	def __init__(self, ballot_factory, markdb, ballot_count, transforms=def_trans, lazy_nondeterminism=True):
		assert isinstance(ballot_factory, CNNScan.Ballot.BallotDefinitions.BallotFactory)
		self.markdb = markdb
		super(GeneratingDataSet, self).__init__(ballot_factory, [ballot_count]*len(ballot_factory), transforms, lazy_nondeterminism=lazy_nondeterminism)

	def load_ballot(self, index, ballot_number):
		# Each contiguous group of `count` ballots is is tfrom the same template
		assert ballot_number >= 0 and ballot_number < len(self.ballot_factory)

		# We shouldn't be asked to create a new ballot when one already exists.
		assert self.marked_ballots[ballot_number][index] is None

		ballot = self.ballot_factory.ballots[ballot_number]

		ret_val = create_marked_ballot(ballot, self.markdb)
		CNNScan.Raster.Raster.crop_contests(ballot, ret_val)
		return ret_val