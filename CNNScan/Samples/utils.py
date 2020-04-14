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

def load_template_image(package, contest:BallotDefinitions.Contest) -> Image:
	a = importlib.resources.open_binary(package, contest.contest_file)
	image = Image.open(a)
	return image