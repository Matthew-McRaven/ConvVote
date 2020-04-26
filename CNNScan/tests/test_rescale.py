# Begin training neural net using default data / parameters
import unittest

import torch
import torchvision
import numpy as np

from CNNScan.Reco import Driver, Settings
import CNNScan.Samples

# Check that padding of all images works, as well as max / average pooling over
# input images.
class TestRescalingOptions(unittest.TestCase):
	def setUp(self):
		# Choose to use real Oregon data (on which the network performs poorly)
		# Or choose randomly generate data, on which the network performs decently.
		self.config = Settings.generate_default_settings()
		self.config['epochs'] = 1
		self.markdb = CNNScan.Mark.MarkDatabase()
		self.markdb.insert_mark(CNNScan.Mark.BoxMark())

		# Create fake data that can be used 
		self.ballot_factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()
		_ = CNNScan.Samples.Random.get_sample_ballot(self.ballot_factory)

		# Create faked marked ballots from ballot factories.
		self.data = CNNScan.Reco.Load.GeneratingDataSet(self.ballot_factory, self.markdb, 2)
		self.load = torch.utils.data.DataLoader(self.data, batch_size=self.config['batch_size'], shuffle=True)
		

	def test_no_rescale(self):
		# Check that rescaling+pooling in the frontend works
		self.config['rescale_pooling'] = False

		# Attempts to train model.
		model = CNNScan.Reco.ImgRec.BallotRecognizer(self.config, self.ballot_factory)
		model = CNNScan.Reco.ImgRec.train_election(model, self.config, self.ballot_factory, self.load, self.load)

	def test_rescale_64(self):
		# Check that average / max pooling on input images to reduce input sizes works
		self.config['rescale_pooling'] = True
		self.config['target_resolution']	= (64, 64)

		# Attempts to train model.
		model = CNNScan.Reco.ImgRec.BallotRecognizer(self.config, self.ballot_factory)
		model = CNNScan.Reco.ImgRec.train_election(model, self.config, self.ballot_factory, self.load, self.load)

	def test_rescale_16(self):
		# Check that average / max pooling on input images to reduce input sizes works
		self.config['rescale_pooling'] = True
		self.config['target_resolution']	= (16, 16)

		# Attempts to train model.
		model = CNNScan.Reco.ImgRec.BallotRecognizer(self.config, self.ballot_factory)
		model = CNNScan.Reco.ImgRec.train_election(model, self.config, self.ballot_factory, self.load, self.load)


if __name__ == '__main__':
    unittest.main()
