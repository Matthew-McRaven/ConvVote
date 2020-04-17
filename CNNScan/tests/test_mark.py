# Begin training neural net using default data / parameters
import unittest

import torch
import torchvision
import numpy as np

from CNNScan.Reco import Driver, Settings
import CNNScan.Samples

class GenerateAllMarks(unittest.TestCase):
	def setUp(self):
		# Choose to use real Oregon data (on which the network performs poorly)
		# Or choose randomly generate data, on which the network performs decently.
		self.config = Settings.generate_default_settings()
		self.config['epochs'] = 1
		self.markdb = CNNScan.Mark.MarkDatabase()
		self.markdb.insert_mark(CNNScan.Mark.BoxMark())
		self.markdb.insert_mark(CNNScan.Mark.InvertMark())
		self.markdb.insert_mark(CNNScan.Mark.XMark())
		self.ballot_factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()
		

	# Check that data can be loaded from the disk
	def test_random(self):
		# Create fake data that can be used 
		data = CNNScan.Reco.Load.GeneratingDataSet(CNNScan.Samples.Random.get_sample_ballot(self.ballot_factory), self.markdb, 10)
		load = torch.utils.data.DataLoader(data, batch_size=self.config['batch_size'], shuffle=True)
		assert len(data.all_ballot_definitions()) == 1
		ballot = data.all_ballot_definitions()[0]
		model = CNNScan.Reco.ImgRec.BallotRecognizer(self.config, ballot)
		model = CNNScan.Reco.ImgRec.train_single_ballot(model, self.config, ballot, load, load)

	# Check that data can be loaded from the disk
	def test_oregon(self):
		# Create fake data that can be used 
		data = CNNScan.Reco.Load.GeneratingDataSet(CNNScan.Samples.Oregon.get_sample_ballot(self.ballot_factory), self.markdb, 10)
		load = torch.utils.data.DataLoader(data, batch_size=self.config['batch_size'], shuffle=True)
		assert len(data.all_ballot_definitions()) == 1
		ballot = data.all_ballot_definitions()[0]
		model = CNNScan.Reco.ImgRec.BallotRecognizer(self.config, ballot)
		model = CNNScan.Reco.ImgRec.train_single_ballot(model, self.config, ballot, load, load)

	# Check that data can be loaded from the disk
	def test_montana(self):
		# Create fake data that can be used 
		data = CNNScan.Reco.Load.GeneratingDataSet(CNNScan.Samples.Montana.get_sample_ballot(self.ballot_factory), self.markdb, 10)
		load = torch.utils.data.DataLoader(data, batch_size=self.config['batch_size'], shuffle=True)
		assert len(data.all_ballot_definitions()) == 1
		ballot = data.all_ballot_definitions()[0]
		model = CNNScan.Reco.ImgRec.BallotRecognizer(self.config, ballot)
		model = CNNScan.Reco.ImgRec.train_single_ballot(model, self.config, ballot, load, load)

if __name__ == '__main__':
    #unittest.main()
	pass
