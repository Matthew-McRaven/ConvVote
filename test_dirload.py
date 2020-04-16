# Begin training neural net using default data / parameters

import torch
import torchvision
import numpy as np

from CNNScan.Reco import Driver, Settings
import CNNScan.Samples


if __name__ == "__main__":
	config = Settings.generate_default_settings()
	config['epochs'] = 1
	# Choose to use real Oregon data (on which the network performs poorly)
	# Or choose randomly generate data, on which the network performs decently.
	transforms=torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda x: np.average(x, axis=-1, weights=[1,1,1,0],returned=True)[0]),
					                           torchvision.transforms.ToTensor(),
											   torchvision.transforms.Lambda(lambda x: x.float()),
											   torchvision.transforms.Normalize((1,),(127.5,))
											   #torchvision.transforms.Lambda(lambda x: (1.0 - (x / 127.5)).float())
											   ])
	data = CNNScan.Reco.Load.DirectoryDataSet("temp/mont",transforms, False)
	load = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True, )

	assert len(data.all_ballot_definitions()) == 1
	ballot = data.all_ballot_definitions()[0]
	model = CNNScan.Reco.ImgRec.BallotRecognizer(config, ballot)
	print(model)
	model = CNNScan.Reco.ImgRec.train_single_ballot(model, config, ballot, load, load)

	# Display a single sample ballot to visualize if training was succesful.
	#render_data = get_test(config, ballot, module)
	#print(render_data)
	#CNNScan.Reco.ImgRec.evaluate_ballots(model, ballot, render_data, config, add_to_ballots=True)

	CNNScan.utils.show_ballot(ballot, data.at(0))