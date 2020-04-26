# Begin training neural net using default data / parameters
import unittest

import torch
import torchvision
import numpy as np

from CNNScan.Reco import Driver, Settings
import CNNScan.Samples

# Our neural net architecture has two output modes.
# If in unique_outputs mode, there is a unique output layer for each (ballot, contest) pair.
# Each output layer has as many neurons as there are options in the associated (ballot, contest) pair.
# Otherwise, there is a single output layer shared between all (ballot, contest) pairs, which is as long as the longest pair.
# This test case ensures that both network configurations work properly.
class TestOutputLabeling(unittest.TestCase):
	def setUp(self):

		self.config = Settings.generate_default_settings()
		self.config['epochs'] = 1
		self.markdb = CNNScan.Mark.MarkDatabase()
		self.markdb.insert_mark(CNNScan.Mark.BoxMark())

		# Create fake data that can be used.
		self.ballot_factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()
		_ = CNNScan.Samples.Random.get_sample_ballot(self.ballot_factory)

		# Create faked marked ballots from ballot factories.
		self.data = CNNScan.Reco.Load.GeneratingDataSet(self.ballot_factory, self.markdb, 2)
		self.load = torch.utils.data.DataLoader(self.data, batch_size=self.config['batch_size'], shuffle=True)


	def test_unique_outputs(self):
		# Configure the NN to have unpooled outputs for each (ballot, contest) pairs.
		self.config['unique_outputs'] = True

		# Attempts to train model.
		model = CNNScan.Reco.ImgRec.BallotRecognizer(self.config, self.ballot_factory)
		model = CNNScan.Reco.ImgRec.train_election(model, self.config, self.ballot_factory, self.load, self.load)

	def test_pooled_outputs(self):
		# Configure the NN to have a pooled output layer shared between all (ballot, contest) pairs.
		self.config['unique_outputs'] = False

		# Attempts to train model.
		model = CNNScan.Reco.ImgRec.BallotRecognizer(self.config, self.ballot_factory)
		model = CNNScan.Reco.ImgRec.train_election(model, self.config, self.ballot_factory, self.load, self.load)


if __name__ == '__main__':
    unittest.main()
