import numpy as np
import torch
import torch.utils.data
import torchvision.transforms

from CNNScan.Reco import ImgRec
from CNNScan.Reco import Settings
import CNNScan.Samples
# Load an election definition file from the disk.
# For now, generates a random election outcome.
def load_ballot_files(config):
	pass

# Create training data using fake ballots.
def get_train(config, ballot, module):
	# TODO: Make normalization a parameter instead of hardcoded.
	transforms=torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda x: np.average(x, axis=-1, weights=[1,1,1,0],returned=True)[0]),
					                           torchvision.transforms.ToTensor(),
											   torchvision.transforms.Lambda(lambda x: x.float()),
											   torchvision.transforms.Normalize((1,),(127.5,))
											   #torchvision.transforms.Lambda(lambda x: (1.0 - (x / 127.5)).float())
											   ])
	data = CNNScan.Mark.mark_dataset(ballot, count=50, transform=transforms)
	#print(data.at(0).marked_contest[0].image)
	load = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True, )
	return load

# Create testing data.
def get_test(config, ballot, module):
	return get_train(config, ballot, module)


# Train a neural network to recognize the results of a single contest for a single election
def contest_entry_point(config, module=CNNScan.Samples.Oregon):
	# TODO: Load real election information from a file.
	ballot = module.get_sample_ballot()
	#config['target_channels'] = 1
	# TODO: scale BallotRecognizer based on election output size
	model = ImgRec.BallotRecognizer(config, ballot)
	print(model)
	model = ImgRec.train_single_ballot(model, config, ballot, get_train(config, ballot, module), get_test(config, ballot, module))

	# Display a single sample ballot to visualize if training was succesful.
	render_data = get_test(config, ballot, module)
	#print(render_data)
	CNNScan.Reco.ImgRec.evaluate_ballots(model, ballot, render_data, config, add_to_ballots=True)

	CNNScan.utils.show_ballot(ballot, render_data.dataset.at(0))
	
	# TODO: write model to file

	# Cleanup memory allocated by Torch
	del model
	del ballot
	torch.cuda.empty_cache()

def main():
	config = Settings.generate_default_settings()
	if True or importlib.util.find_spec("ray") is None:
		contest_entry_point(config)
	else:
		from ray import tune
		import ray
		ray.init()
		config['cuda'] = False
		# TODO: Log accuracy results within neural network.
		analysis = tune.run(contest_entry_point, config=config, resources_per_trial={ "cpu": 1, "gpu": 0.0})
# Launch a training run, with optional hyperparameter sweeping.
if __name__ == "__main__":
	main()