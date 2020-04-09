import torch

from CNNScan.Reco import ImgRec
from CNNScan.Reco import Settings
import CNNScan.Samples

# Load an election definition file from the disk.
# For now, generates a random election outcome.
def load_ballot_files(config):
	pass

# Create training data using fake ballots.
def get_train(config, ballot, module):
	#marked_ballots = ElectionFaker.create_fake_marked_ballots(ballot, 400)
	marked_ballots = CNNScan.Samples.utils.make_sample_ballots(module, ballot, count=100)
	n = config['batch_size']
	return [marked_ballots[i * n:(i + 1) * n] for i in range((len(marked_ballots) + n - 1) // n )]

# Create testing data.
def get_test(config, ballot, module):
	return get_train(config, ballot, module)


# Train a neural network to recognize the results of a single contest for a single election
def contest_entry_point(config, module=CNNScan.Samples.Oregon):
	# TODO: Load real election information from a file.
	""" 
	Using CNNScan.Samples.Random will generate completely random ballots marked with black bars.
	Using CNNScan.Samples.Oregon will attempt to fill in real Oregon ballots correctly.
	"""
	# module = CNNScan.Samples.Oregon 
	# module = CNNScan.Samples.Random 
	ballot = module.get_sample_ballot()
	#config['target_channels'] = 1
	# TODO: scale BallotRecognizer based on election output size
	model = ImgRec.BallotRecognizer(config, ballot)
	print(model)
	model = ImgRec.train_single_ballot(model, config, ballot, get_train(config, ballot, module), 
	 get_test(config, ballot, module))
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