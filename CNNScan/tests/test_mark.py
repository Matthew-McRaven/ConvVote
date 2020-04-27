import unittest

import torch
import torchvision
import numpy as np

from CNNScan.Reco import Settings
import CNNScan.Samples

# Validate that all marking classes can be applied to each and every ballot.
class GenerateAllMarks(unittest.TestCase):
	def setUp(self):
		# Choose to use real Oregon data (on which the network performs poorly)
		# Or choose randomly generate data, on which the network performs decently.
		self.config = Settings.generate_default_settings()
		self.config['epochs'] = 1
		# Reduce size of NN to speed up testing.
		self.config['recog_full_layers'] = [10]
		self.config['recog_conv_layers'] 	= [
		CNNScan.Settings.conv_def(4, 1, 4, 0, 4, False),
		]
		self.markdb = CNNScan.Mark.MarkDatabase()
		self.markdb.insert_mark(CNNScan.Mark.BoxMark())
		self.markdb.insert_mark(CNNScan.Mark.InvertMark())
		self.markdb.insert_mark(CNNScan.Mark.XMark())
		

	# Check that random dataset works correctly.
	def test_random(self):
		# Create fake data that can be used 
		ballot_factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()
		_ = CNNScan.Samples.Random.get_sample_ballot(ballot_factory, 30)

		# Create faked marked ballots from ballot factories.
		data = CNNScan.Reco.Load.GeneratingDataSet(ballot_factory, self.markdb, 2)
		load = torch.utils.data.DataLoader(data, batch_size=self.config['batch_size'], shuffle=True)

		# Attempts to train model.
		model = CNNScan.Reco.ContestRec.BallotRecognizer(self.config, ballot_factory)
		model = CNNScan.Reco.ContestRec.train_election(model, self.config, ballot_factory, load, load)

	# Check that Oregon dataset works correctly.
	def test_oregon(self):
		# Create fake data that can be used 
		ballot_factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()
		_ = CNNScan.Samples.Oregon.get_sample_ballot(ballot_factory, 30)

		# Create faked marked ballots from ballot factories.
		data = CNNScan.Reco.Load.GeneratingDataSet(ballot_factory, self.markdb, 2)
		load = torch.utils.data.DataLoader(data, batch_size=self.config['batch_size'], shuffle=True)

		# Attempts to train model.
		model = CNNScan.Reco.ContestRec.BallotRecognizer(self.config, ballot_factory)
		model = CNNScan.Reco.ContestRec.train_election(model, self.config, ballot_factory, load, load)

	# Check that Montana dataset works correctly.
	def test_montana(self):
		# Create fake data that can be used 
		ballot_factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()
		_ = CNNScan.Samples.Montana.get_sample_ballot(ballot_factory, 30)

		# Create faked marked ballots from ballot factories.
		data = CNNScan.Reco.Load.GeneratingDataSet(ballot_factory, self.markdb, 2)
		load = torch.utils.data.DataLoader(data, batch_size=self.config['batch_size'], shuffle=True)

		# Attempts to train model.
		model = CNNScan.Reco.ContestRec.BallotRecognizer(self.config, ballot_factory)
		model = CNNScan.Reco.ContestRec.train_election(model, self.config, ballot_factory, load, load)

if __name__ == '__main__':
    #unittest.main()
	pass
