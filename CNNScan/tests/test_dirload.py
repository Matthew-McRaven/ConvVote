# Begin training neural net using default data / parameters
import unittest

import torch
import torchvision
import numpy as np

from CNNScan.Reco import Driver, Settings
import CNNScan.Samples

class TestLoadDisk(unittest.TestCase):
	def setUp(self):
		self.config = Settings.generate_default_settings()
		self.config['epochs'] = 1

	# Check that data can be loaded from the disk
	def test_load_one(self):
		return
		# Create factory, create one ballot
		factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()
		ballot = factory.Ballot(CNNScan.Samples.Oregon.contests, ballot_file=CNNScan.Samples.Oregon.ballot_file)

		# Rasterize ballots to disk
		CNNScan.Mark.main(factory=factory, dpi=100, outdir="temp/test00", count=5)

		# Create data loaders for saved data
		data = CNNScan.Reco.Load.DirectoryDataSet("temp/test00",CNNScan.Reco.Load.def_trans, False)
		load = torch.utils.data.DataLoader(data, batch_size=self.config['batch_size'], shuffle=True)

		# Train model on data
		model = CNNScan.Reco.ImgRec.BallotRecognizer(self.config, factory)
		model = CNNScan.Reco.ImgRec.train_election(model, self.config, factory, load, load)

	# Check that many ballots may be written to the
	def test_load_many(self):
		# Create factory, create multiple ballots.
		factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()
		_ =[factory.Ballot(CNNScan.Samples.Oregon.contests, ballot_file=CNNScan.Samples.Oregon.ballot_file) for i in range(3)]

		# Rasterize ballots to disk.
		CNNScan.Mark.main(factory=factory, dpi=100, outdir="temp/test01", count=5)

		# Create data loaders for saved data. Save to different directory than previous tests.
		data = CNNScan.Reco.Load.DirectoryDataSet("temp/test01",CNNScan.Reco.Load.def_trans, False)
		load = torch.utils.data.DataLoader(data, batch_size=self.config['batch_size'], shuffle=True)

		# Train model on data.
		model = CNNScan.Reco.ImgRec.BallotRecognizer(self.config, factory)
		model = CNNScan.Reco.ImgRec.train_election(model, self.config, factory, load, load)

if __name__ == '__main__':
	unittest.main()
