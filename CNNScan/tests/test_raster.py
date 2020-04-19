import typing
import os

import numpy
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from PIL import Image
import matplotlib.pyplot as plt

from CNNScan.Raster import Raster
import CNNScan.Samples.utils
import CNNScan.Samples.Oregon
import CNNScan.Reco.Load
from CNNScan.Ballot import BallotDefinitions, MarkedBallots
import CNNScan.Mark
import CNNScan.Reco
import unittest

import torch
import torchvision
import numpy as np

from CNNScan.Reco import Driver, Settings
import CNNScan.Samples

class TestConvertPDF2Image(unittest.TestCase):
		

	# Check that data can be loaded from the disk
	def test_raster(self):
		oregon = CNNScan.Samples.Oregon
		ballot_factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()
		ballot = ballot_factory.Ballot(contests=oregon.contests, ballot_file=oregon.ballot_file)

		ballot = CNNScan.Raster.Raster.rasterize_ballot_image(ballot , 400)
		#print(value)
		config = CNNScan.Reco.Settings.generate_default_settings()
		# Display a single sample ballot to visualize if training was succesful.
		render_data = CNNScan.Reco.Driver.get_test(config, ballot_factory)

		#CNNScan.utils.show_ballot(render_data.dataset.ballot_definition(0), render_data.dataset.at(0))
		"""for i, contest in enumerate(render_data.dataset.at(0).marked_contest):
			print(contest.index)
			if contest.index in [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]:
				continue
			fig = plt.figure()
			ax = fig.add_subplot( 1, 1, 1)
			ax.set_title(f'Contest {contest.index}')
			ax.set_xlabel(f'Vote for {contest.actual_vote_index}. Recorded as {contest.computed_vote_index}')
			ax.imshow(contest.image, interpolation='nearest')
		plt.show()"""

if __name__ == '__main__':
    unittest.main()