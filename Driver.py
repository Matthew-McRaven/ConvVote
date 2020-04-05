import ElectionFaker as ElectionFaker
import ImgRec as ImgRec
import settings as Settings
import torch

# Load an election definition file from the disk.
# For now, generates a random election outcome.
def load_ballot_files(config):
	ballot = ElectionFaker.create_fake_ballot()
	return ballot

# Create training data using fake ballots.
def get_train(config, ballot):
	marked_ballots = ElectionFaker.create_fake_marked_ballots(ballot, 400)
	n = config['batch_size']
	return [marked_ballots[i * n:(i + 1) * n] for i in range((len(marked_ballots) + n - 1) // n )]

# Create testing data.
def get_test(config, ballot):
	return get_train(config, ballot)


# Train a neural network to recognize the results of a single contest for a single election
def contest_entry_point(config):
	# TODO: Load real election information from a file.
	ballot = load_ballot_files(config)
	# TODO: scale BallotRecognizer based on election output size
	model = ImgRec.BallotRecognizer(config, ballot)
	model = ImgRec.train_single_ballot(model, config, ballot, get_train(config, ballot), 
	 get_test(config, ballot))
	# TODO: write model to file

	# Cleanup memory allocated by Torch
	del model
	del ballot
	torch.cuda.empty_cache()


# Launch a training run, with optional hyperparameter sweeping.
if __name__ == "__main__":
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