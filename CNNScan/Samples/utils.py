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
	return image

def get_sample_dataset(module, ballot:BallotDefinitions.Ballot, count=100, transform=None):
	mark_db = CNNScan.Mark.MarkDatabase()
	mark_db.insert_mark(CNNScan.Mark.XMark())
	return CNNScan.Reco.Load.ImageDataSet(ballot, mark_db, count, transform=transform)