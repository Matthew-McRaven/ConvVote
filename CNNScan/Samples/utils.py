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

def get_sample_dataset(module, ballot:BallotDefinitions.Ballot, count=100, transform=None):
	mark_db = CNNScan.Mark.MarkDatabase()
	mark_db.insert_mark(CNNScan.Mark.XMark())
	return CNNScan.Reco.Load.ImageDataSet(ballot, mark_db, count, transform=transform)