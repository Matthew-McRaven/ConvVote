# This is where our conv neural net model will go.

import time, os, sys, random, datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import collections
import typing

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import PIL
from PIL import Image

import CNNScan.Reco.Settings as Settings
import CNNScan.utils as utils
from CNNScan.Ballot import BallotDefinitions, MarkedBallots
import CNNScan.utils
import math

class ImageRecognitionCore(nn.Module):
	def __init__(self, config, input_dimensions):
		super(ImageRecognitionCore, self).__init__()
		self.input_dimensions = input_dimensions

		# Construct convolutional layers.
		H = input_dimensions[0]
		W = input_dimensions[1]
		conv_list = []
		non_linear = Settings.get_nonlinear(config['recog_conv_nlo'])
		self.in_channels = in_channels = config['target_channels']

		# Iterate over all pooling/convolutional layer configurations.
		# Construct all items as a (name, layer) tuple so that the layers may be loaded into
		# an ordered dictionary. Ordered dictionaries respect the order in which items were inserted,
		# and are the least painful way to construct a nn.Sequential object.
		for index, item in enumerate(config['recog_conv_layers']):
			# Next item is a convolutional layer, so construct one and re-compute H,W, channels.
			if isinstance(item, Settings.conv_def):
				conv_list.append((f'conv{index}', nn.Conv2d(in_channels, item.out_channels, item.kernel,
				 stride=item.stride, padding=item.padding, dilation=item.dilation)))
				H = utils.resize_convolution(H, item.kernel, item.dilation, item.stride, item.padding)
				W = utils.resize_convolution(W, item.kernel, item.dilation, item.stride, item.padding)
				in_channels = item.out_channels
			# Next item is a pooling layer, so construct one and re-compute H,W.
			elif isinstance(item, Settings.pool_def):
				if item.pool_type.lower() == 'avg':
					conv_list.append((f'avgpool{index}',nn.AvgPool2d(item.kernel, stride=item.stride, padding=item.padding)))
					H = utils.resize_convolution(H, item.kernel, 1, item.stride, item.padding)
					W = utils.resize_convolution(W, item.kernel, 1, item.stride, item.padding)
				elif item.pool_type.lower() == 'max':
					conv_list.append((f'maxpool{index}', nn.MaxPool2d(item.kernel, stride=item.stride, padding=item.padding, dilation=item.dilation)))
					H = utils.resize_convolution(H, item.kernel, item.dilation, item.stride, item.padding)
					W = utils.resize_convolution(W, item.kernel, item.dilation, item.stride, item.padding)
				else:
					raise NotImplementedError(f"{item.pool_type.lower()} is not an implemented form of pooling.")
			self.output_layer_size = H * W * in_channels
			# Add a non-linear operator if specified by item. Non linear operators also pair with dropout
			# in all the examples I've seen
			if item.non_linear_after:
				conv_list.append((f"{config['recog_conv_nlo']}{index}", non_linear))
				conv_list.append((f"dropout{index}", nn.Dropout(config['dropout'])))
		
		# Group all the convolutional layers into a single callable object.
		self.conv_layers = nn.Sequential(collections.OrderedDict(conv_list))
		
		# Randomize initial parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.kaiming_normal_(p)

	def forward(self, _, contest_number, batches, images):

		H,W = self.input_dimensions
		# Magic line of code needed to cast image vector into correct dimensions?
		images = images.view(batches, self.in_channels, H, W)
		outputs = self.conv_layers(images)
		return outputs

	# Return the number of dimensions in the output size
	def output_size(self):
		return self.output_layer_size


class ImageRescaler(nn.Module):
	def __init__(self, config, ballot_factory):
		super(ImageRescaler, self).__init__()
		# Configuration gives us a target output resolution.
		
		# Don't allow an output size that is not a power of 2.
		self.use_pool = config['rescale_pooling']
		# If pooling is enabled, output resolutions must be a power of 2.
		if self.use_pool:
			self.x_out_res, self.y_out_res = config['target_resolution']
			assert(utils.is_power2(self.x_out_res))
			assert(utils.is_power2(self.y_out_res))
		else:
			# Otherwise, x,y only need to be padded to the size of the largest x,y size ever seen.
			self.x_out_res, self.y_out_res = 0,0
			for ballot in ballot_factory.ballots:
				for contest in ballot.contests:
					imagex = contest.abs_bounding_rect.lower_right.x - contest.abs_bounding_rect.upper_left.x
					imagey = contest.abs_bounding_rect.lower_right.y - contest.abs_bounding_rect.upper_left.y
					self.x_out_res = max(self.x_out_res, imagex)
					self.y_out_res = max(self.y_out_res, imagey)
		#print(self.x_out_res, self.y_out_res)


		# Create a 2d array (filled in by resize) that maps a (ballot, contest) to an offset in the padding and pooling arrays.
		self.translate = []
		self.pad = nn.ModuleList()
		self.pool = nn.ModuleList()
		# Input resolutions expected at the i'th entry of pad, pool.
		self.x_in_res, self.y_in_res = [], []
		# Fill in translation, pad, pool arrays.
		self.resize(ballot_factory)

	def forward(self, ballot_number, contest_number, batches, images):
		out_tens =[]
		#print(ballot_number, contest_number, images)
		assert len(ballot_number) == len(contest_number)
		assert len(contest_number) == len(images)
		assert len(images) == batches
		# Each image may be a different size and may come from a different (ballot, contest) tuple
		# Therefore, we can't truly batch this operation.
		for index in range(len(ballot_number)):
			ballot = ballot_number[index]
			contest = contest_number[index]
			lookup_index = self.translate[ballot][contest]
			# TODO: Perform pooling, interpolation, and nothing depending on the size ratios of in to out resolution.
			out = self.pad[lookup_index](images[index])
			#print(out.shape)
			if self.use_pool:
				out = self.pool[lookup_index](out)
			out_tens.append(out)
		# All output tensors are now the same size.
		output = torch.stack(out_tens)
		return output

	def output_size(self):
		return self.x_out_res * self.y_out_res

	# Return raw dimensions of output images, needed when performing convolutional filtering
	# in image recognition core.
	def output_dimensions(self):
		return (self.x_out_res, self.y_out_res)

	def resize(self, ballot_factory):
		assert isinstance(ballot_factory, BallotDefinitions.BallotFactory)
		# Create a 2d array to map from (ballot, contest) to offsets in padding, pooling arrays
		self.translate = len(ballot_factory.ballots) * [None]
		for outer, ballot in enumerate(ballot_factory.ballots):
			start = len(self.pad)
			self.translate[outer] = [i+start for i in range(len(ballot.contests))]
			for inner, contest in enumerate(ballot.contests):
				# Determine the height, width of each image by subtracting the bounding rectangles from each other.
				imagex = contest.abs_bounding_rect.lower_right.x - contest.abs_bounding_rect.upper_left.x
				imagey = contest.abs_bounding_rect.lower_right.y - contest.abs_bounding_rect.upper_left.y
				if self.use_pool:
					# Must check that image being passed in is a power of 2, else pooling will fail.
					x_in_res, pad_left, pad_right = utils.pad_nearest_pow2(imagex, self.x_out_res)
					y_in_res, pad_top, pad_bottom = utils.pad_nearest_pow2(imagey, self.y_out_res)
					#print(imagex, imagey)
					# Pad with 0's
					#
					# Since we've picked input resolutions, output resolutions that are powers of 2,
					# ratios should be whole numbers (even without rounding).
					x_ratio, y_ratio = x_in_res//self.x_out_res, y_in_res//self.y_out_res
					self.pool.append(nn.AvgPool2d((x_ratio, y_ratio)))
				else:
					x_padding, y_padding = self.x_out_res - imagex, self.y_out_res - imagey
					pad_left = x_padding // 2
					pad_right = x_padding - pad_left
					pad_top = y_padding // 2
					pad_bottom = y_padding - pad_top

				# Print the number of entries padded on each side of the image.
				#print(pad_left, pad_right, pad_bottom, pad_top)

				# For now, only pad with zeros.
				# Eventually, it would be nice to pad with 1's.
				if True:
					self.pad.append(nn.ZeroPad2d((pad_left, pad_right, pad_bottom, pad_top)))
				# Pad with Green
				elif False:
					self.pad.append(nn.ConstantPad3d((pad_left, pad_right, pad_bottom, pad_top,0,0), 255))
				# Pad with replication
				else:
					self.pad.append(nn.ReplicationPad2d((pad_left, pad_right, pad_bottom, pad_top)))

				# Need input resolution to properly view(...) input tensor
				self.x_in_res.append(imagex)
				self.y_in_res.append(imagey)
		# Print out the table that converts from 2d (ballot, contest) to the 1d module objects.
		print(self.translate)

		# Randomize initial parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.kaiming_normal_(p)

# A class that takes an arbitrary number of inputs and useses it to perform class recognition on the data.
# The output dimension should match the number of classes in the data.
class OutputLabeler(nn.Module):
	def __init__(self, config, input_dimensions, ballot_factory):
		super(OutputLabeler, self).__init__()
		self.config = config
		# Not all network architectures / loss functions need outputs mapped onto [0,1].
		# TODO: Read setting from config file
		self.use_sigmoid = False
		self.input_dimensions = input_dimensions
		# Create an embedding table to "learn" how a (ballot#, contest#) should be interpreted.
		self.tx_table = nn.Embedding(ballot_factory.num_contests(), config['recog_embed'])

		# We concat the input with entries from our embedding table, so the "starting" size must be augmented.
		last_size = self.input_dimensions+config['recog_embed']
		# Get the non-linear operator to be used in the fully connected layers.
		non_linear = Settings.get_nonlinear(config['recog_full_nlo'])

		# Iterate over list of fully connected layer definitions.
		# Construct all items as a (name, layer) tuple so that the layers may be loaded into
		# an ordered dictionary. Ordered dictionaries respect the order in which items were inserted,
		# and are the least painful way to construct a nn.Sequential object.
		layer_list =[]
		for index, layer_size in enumerate(config['recog_full_layers']):
			layer_list.append((f"fc{index}", nn.Linear(last_size, layer_size)))
			layer_list.append((f"{config['recog_full_nlo']}{index}", non_linear))
			layer_list.append((f"dropout{index}", nn.Dropout(config['dropout'])))
			last_size = layer_size

		self.linear = nn.Sequential(collections.OrderedDict(layer_list)) #nn.Linear(input_dimensions, self.output_layer_size)

		self.output_dimension = []
		if not self.config['unique_outputs']:
			self.output_layers=nn.Linear(last_size, ballot_factory.max_options())
			self.output_dimension=ballot_factory.max_options()
		else:
			#self.translate = [None] * len(ballot_factory.ballots)
			self.output_layers = nn.ModuleList()
			for _, ballot in enumerate(ballot_factory.ballots):
				for contest in ballot.contests:
					this_dim = len(contest.options)
					self.output_dimension.append(this_dim)
					layer_list = []
					# TODO: Add switch to enable private linear layer or force sharing of all FC layers.
					#layer_list.append(("private-linear", nn.Linear(last_size, 100)))
					#layer_list.append(("Output", nn.Linear(100, this_dim)))
					layer_list.append(("Output", nn.Linear(last_size, this_dim)))
					self.output_layers.append(nn.Sequential(collections.OrderedDict(layer_list)))	

		self.sigmoid = nn.Sigmoid()

		# Create a table that maps from (ballot#, contest#) to an offset in the embedding table by enumerating all ballot definitons.
		start = 0
		self.index_of = len(ballot_factory.ballots) * [None]
		for outer, ballot in enumerate(ballot_factory.ballots):
			self.index_of[outer] = [i+start for i in range(len(ballot.contests))]
			start += len(self.index_of[outer])

		# Randomize initial parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.kaiming_normal_(p)
	
	def forward(self, ballot_number, contest_number, batches, inputs):
		# Input may be coming from CNN, so we must flatten to 1 dimension.
		inputs = inputs.view(batches, -1)
		# Convert from the 2-d coordinates of (ballot, contest) to offset in embedding table.
		wanted_embeds = []
		for index in range(len(ballot_number)):
			ballot = ballot_number[index]
			contest = contest_number[index]
			wanted_embeds.append(self.index_of[ballot][contest])
		# Create a tensor that can batch-index the embedding table
		tens = utils.cuda(torch.tensor(wanted_embeds, dtype=torch.long), self.config)
		# Get all entries from the embedding table, and move it to the GPU.
		emb = utils.cuda(self.tx_table(tens), self.config)
		# Merge the inputs and embedding definitions into a single quantity that can be manipulated by the linear layers.
		inputs = torch.cat((emb, inputs), 1)
		output = self.linear(inputs)
		if self.config['unique_outputs']:
			stackable = []
			for index in range(len(ballot_number)):
				ballot = ballot_number[index]
				contest = contest_number[index]
				stackable.append(self.output_layers[self.index_of[ballot][contest]](output[index]))
			output = torch.stack(stackable)
		else:
			output = self.output_layers(output)

		# Not all network architectures need a sigmoid applied to the output, so this needs to be configurable.
		if self.use_sigmoid:
			# Map outputs into range [0,1], with 1 being a vote for an option
			# and 0 being the absence of a vote.
			output = self.sigmoid(output)

		return output

# A container that holds the three portions of our Neural Network.
# The first segment rescales the images to be a standard size (which is input_image_x,input_image_y).
# The second segment uses a CNN to perform image recognition on the input image.
# The third segment selects the best fitting label for that data presented to it.
class BallotRecognizer(nn.Module):
	def __init__(self, config, ballot_factory):
		super(BallotRecognizer, self).__init__()
		assert isinstance(ballot_factory, BallotDefinitions.BallotFactory)
		rescaler = ImageRescaler(config, ballot_factory)
		recognizer = ImageRecognitionCore(config, rescaler.output_dimensions())
		labeler = OutputLabeler(config, recognizer.output_size(), ballot_factory)
		self.dump_idx = 0

		# Use an ordered dict so printing out the model prints in the correct order.
		self.module_list = nn.ModuleDict(collections.OrderedDict([
			('rescaler', rescaler),
			('recognizer', recognizer),
			('labeler', labeler)])
			)

	reset =  torchvision.transforms.Compose([# Genius 1-liner to undo-normalization from:
											 # 		https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/5
					                         torchvision.transforms.Normalize((-1/127.5,),(1/127.5,)),
											 torchvision.transforms.ToPILImage(),
											])	
	def forward(self, ballot_number, contest_number, inputs):
		batches = len(inputs)
		outputs = self.module_list['rescaler'](ballot_number, contest_number, batches, inputs)
		if True:
			rep = outputs[0]
			im = self.reset(rep).convert("RGB")
			os.makedirs(f"temp/imdump/", exist_ok=True)
			im.save(f"temp/imdump/{self.dump_idx}.png")
			im.close()
			self.dump_idx+=1
		outputs = self.module_list['recognizer'](ballot_number, contest_number, batches, outputs)
		outputs = self.module_list['labeler'](ballot_number, contest_number, batches, outputs)
		outputs = outputs.view(batches, -1)
		return outputs
	

def train_single_contest(model, config, train_data, test_data, number_candidates):
	raise NotImplementedError()

def train_election(model, config, ballot_factory, train_loader, test_loader):
	if config['cuda']:
		model = utils.cuda(model, config)
	
	# Create loss function(s) for task.
	criterion = Settings.get_criterion(config)

	# Choose an optimizer.
	optimizer = Settings.get_optimizer(config, model)

	# Train the network for a given number of epochs.
	for epoch in range(config['epochs']):

		
		print(f"\n\nEpoch {epoch}")
		print("From 0 wrong to n wrong.")
		print("None selected, idx0 to idxn")

		model.train()
		optimizer.zero_grad()

		# batch_images contains the # of images evaluated in the last batch
		# batch_loss is the summed magnitude of total losst.
		# batch_correct is a [i][j] array.
		# The i'th dimension is a summary over all contests with i+1 options.
		# The j dimensions corresponds to how many mistakes were made, with j being the # of mistakes.
		# e.g. batch_correct[6][5] returns the number of times a contest with length 7 was evaluated with 4 errors.
		# batch_select is an [i][j] array recording the distribution of votes.
		# batch_select[i][0 corresponds to the number of contests with i-1 options wehre NO OPTION WAS VOTED FOR.
		# batch_select[i][j] records the number of times the j-1'th option was selected in a contest with i-1 options
		(batch_images, batch_loss, batch_correct, batch_select) = iterate_loader_once(config, model, ballot_factory, train_loader, criterion=criterion, optimizer=optimizer, train=True, annotate_ballots=True)
		
		#print(f"Guessed {batch_correct} options out of {batch_images} total for {100*batch_correct[0]/batch_images}% accuracy. Loss of {batch_loss}.")
		
		model.eval()
		count = 0
		correct = 0
		with torch.no_grad():
			(batch_images, batch_loss, batch_correct, batch_select) = iterate_loader_once(config, model, ballot_factory, test_loader, criterion=criterion, train=False)
			for i, row in enumerate(batch_correct):
				if sum(row) == 0: # If there were no contests with i options, skip.
					continue
				print(f"Accuracy with {i+1} options is: {row[0:i+2]}")
				print(f"Selection distribution is: {batch_select[i][0:i+2]}\n")
				correct+=row[0]
				count+=sum(row)

		print(f"Accuracy of {100*correct/count}% for {count} contests")

	return model


# Iterate over all the data in a loader one time.
# Account for varying number of ballots, as well annotating the data in the data loader with recorded votes.
# Works for both training and development. Will probably not work with real data, since no labels/loss will be available.
def iterate_loader_once(config, model, ballot_factory, loader, criterion=None, optimizer=None, train=True, annotate_ballots=True, count_options=False):
	batch_images, batch_loss, batch_correct = 0,0,[None]*ballot_factory.max_options()
	# The [i][j]'th entry of batch select indicates for a contest with n options, how many votes there were for the j-1th option.
	# Note that [i][0] records the number of "no vote for this contest". Also sum(array[i]) =/= number of ballots.
	batch_select = [None]*(ballot_factory.max_options())
	# Initialize the second dimension of the arrays. 
	# Can't use a compound list comprehension, as it caused all inner lists to point to the same object instance (aka the same array).
	for i in range(ballot_factory.max_options()):
		batch_correct[i]=[0]*(ballot_factory.max_options()+1)
	for i in range(ballot_factory.max_options()):
		batch_select[i]=[0]*(ballot_factory.max_options()+1)

	# Randomize the order that we visit ballots in, hopefully reducing bias in training.
	# Ultimately, it would be best to intermix contests from all ballots, but our data loader cannot
	# handle variation within sets of image tensors
	ballot_types = [i for i in range(len(ballot_factory))]
	random.shuffle(ballot_types)
	for ballot_type in ballot_types:
		# Make the len() and [] calls to the data loader resolve to a single ballot type,
		# rather than accessing all ballot types at will.
		loader.dataset.freeze_ballot_definiton_index(ballot_type)

		for dataset_index, ballot_numbers, labels, images in loader:
			# Create an array containing the ballot indecies of the i'th data item.
			# (A ballot index points you to a specific ballot definition in the ballot_factory)
			# Must be array, so that inputs to model may mix multiple contests from multiple ballots in a single batch.
			ballot_numbers = ballot_numbers.type(torch.LongTensor)

			# Shuffle the order in which contests are visited.
			contest_numbers = [i for i in range(len(ballot_factory.ballots[ballot_type].contests))]
			random.shuffle(contest_numbers)

			for contest_idx in contest_numbers:
				tensor_images = utils.cuda(images[contest_idx], config)
				tensor_labels = utils.cuda(labels[contest_idx], config)
				# Must move tensor to cuda *after* converting it to floats.
				#tensor_labels = utils.cuda(tensor_labels.type(torch.FloatTensor), config)
				# Create an array containing the contest indecies of the i'th data item.
				# (A contest index points you to a contest in a given ballot definition)
				# Must be array, so that inputs to model may mix multiple contests from multiple ballots in a single batch.
				tensor_contest_idx = utils.cuda(torch.full( (len(dataset_index),), contest_idx, dtype=torch.long), config)
				output = model(ballot_numbers, tensor_contest_idx, tensor_images)

				# Output has as many dimensions as contest with the most options for all ballots in the ballot factory.
				# We don't care about outputs of the network that don't have corresponding labels, so chop them off.
				# e.g. the output may have length 7, but the current ballot may only have 2 options.
				# TODO: Mixed contests/ballots must resize the dimension iteratively.
				#output = output.narrow(-1, 0, tensor_labels.shape[-1])

				loss = criterion(output, tensor_labels)

				# Perform pytorch training.
				if train:
					loss.backward()
					optimizer.step()

				# Annotate ballots with the list of recorded votes.				
				# Compute the number of options determined correctly
				batch_images+=len(tensor_images)
				for (index, output_labels) in enumerate(output):
					#print(output_labels)
					val, selected_idx = output_labels.max(0)
					batch_select[len(output_labels) -1][selected_idx+1]+=1
					if  selected_idx == tensor_labels[index]:
						batch_correct[len(output_labels)-1][0]+=1
					else:
						batch_correct[len(output_labels)-1][1]+=1

				# Accumulate losses
				batch_loss += loss.data.item()

				# Clean up memory, since CUDA seems to leak memory when running for a long time.
				del tensor_images
				del tensor_labels
				del loss
				del output
				torch.cuda.empty_cache()

	return (batch_images, batch_loss, batch_correct, batch_select)
