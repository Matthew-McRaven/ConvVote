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
def get_train(config, ballot_factory):
	mark_db = CNNScan.Mark.MarkDatabase()
	mark_db.insert_mark(CNNScan.Mark.BoxMark()) 
	data = CNNScan.Reco.Load.GeneratingDataSet(ballot_factory, mark_db, 50)
	#data = CNNScan.Reco.Load.GeneratingDataSet(ballot, mark_db, 100, transforms)
	#print(data.at(0).marked_contest[0].image)
	load = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True, )
	return load

# Create testing data.
def get_test(config, ballot):
	return get_train(config, ballot)


# Train a neural network to recognize the results of a single contest for a single election
def contest_entry_point(config, module=CNNScan.Samples.Oregon):
	# TODO: Load real election information from a file.
	ballot_factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()
	ballot = module.get_sample_ballot(ballot_factory)
	#config['target_channels'] = 1
	# TODO: scale BallotRecognizer based on election output size
	model = ImgRec.BallotRecognizer(config, ballot)
	print(model)
	model = ImgRec.train_single_ballot(model, config, ballot, get_train(config, ballot_factory), get_test(config, ballot_factory))

	# Display a single sample ballot to visualize if training was succesful.
	render_data = get_test(config, ballot_factory)
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