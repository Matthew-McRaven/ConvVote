# Begin training neural net using default data / parameters
import unittest

import torch
import torchvision
import numpy as np

from CNNScan.Reco import Driver, Settings
import CNNScan.Samples

class TestLoadDisk(unittest.TestCase):
	def setUp(self):
		#raise NotImplementedError("Fuck you")
		# Choose to use real Oregon data (on which the network performs poorly)
		# Or choose randomly generate data, on which the network performs decently.
		self.config = Settings.generate_default_settings()
		self.config['epochs'] = 1
		# Create fake data that can be used 
		factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()
		ballot = factory.Ballot(CNNScan.Samples.Oregon.contests, ballot_file=CNNScan.Samples.Oregon.ballot_file)
		CNNScan.Mark.main(factory=factory, dpi=100, outdir="temp/test", count=5)
		self.data = CNNScan.Reco.Load.DirectoryDataSet("temp/test",CNNScan.Reco.Load.def_trans, False)
		self.load = torch.utils.data.DataLoader(self.data, batch_size=self.config['batch_size'], shuffle=True)

	# Check that data can be loaded from the disk
	def test_training(self):
		assert len(self.data.all_ballot_definitions()) == 1
		ballot = self.data.all_ballot_definitions()[0]
		model = CNNScan.Reco.ImgRec.BallotRecognizer(self.config, ballot)
		model = CNNScan.Reco.ImgRec.train_single_ballot(model, self.config, ballot, self.load, self.load)

if __name__ == '__main__':
	unittest.main()
	pass
