import ElectionFaker as ElectionFaker
import ImgRec as ImgRec
import settings as Settings
import torch

# Load an election definition file from the disk.
# For now, generates a random election outcome.
def load_election_files(config):
	election = ElectionFaker.create_election()
	return election

# Create training data using fake ballots.
def get_train(config, contest, contest_phys_data):
	marked_ballots = ElectionFaker.create_fake_marked_ballots(contest, contest_phys_data, 400)
	n = config['batch_size']
	return [marked_ballots[i * n:(i + 1) * n] for i in range((len(marked_ballots) + n - 1) // n )]

# Create testing data.
def get_test(config, contest, contest_phys_data):
	return get_train(config, contest, contest_phys_data)


# Train a neural network to recognize the results of a single contest for a single election
def contest_entry_point(config):
	# TODO: Load real election information from a file.
	election = load_election_files(config)
	contest, contest_phys = ElectionFaker.create_fake_contest()
	# TODO: scale BallotRecognizer based on election output size
	model = ImgRec.BallotRecognizer(config, contest_phys.bound_rect[2], contest_phys.bound_rect[3])
	model = ImgRec.train_single_contest(model, config, get_train(config, contest, contest_phys), get_test(config, contest, contest_phys))
	# TODO: write model to file

	# Cleanup memory allocated by Torch
	del model
	del election
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