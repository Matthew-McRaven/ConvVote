import torch
import torch.nn as nn

# Create sensible defaults for all the settings in the network.
def generate_default_settings():
	ret = {}
	# Global configuration
	# Determine globally if CUDA should be enabled or disabled.
	ret['cuda'] 			= torch.cuda.is_available()
	# Use faked data for election?
	ret['fake'] 			= True
	# Otherwise, load the following pickled file as an election result.
	ret['election_file'] 	= None

	# Logging configuration parameters
	# After this many messages, log training, testing progress.
	ret['train_log_int']	= 1000

	# See Settings::get_criterion() for available criterion.
	ret['criterion'] 		= 'MSE-SUM'
	# See Settings::get_criterion() for available criterion.
	ret['optimizer'] 		= 'adam'

	# Model hyperparameters
	ret['learning_rate'] 	= 0.01
	ret['l2_lambda'] 		= 0.01
	ret['dropout'] 			= 0.1
	ret['epochs'] 			= 10
	ret['batch_size']		= 20

	# Determine how to pad / cut the images
	ret['target_resolution']= (64, 64) 
	# Determine how the inner CNN / FC layers are built
	# How many values should come out of the ImageRecognitionCore class?
	ret['recog_out_dim']	= 100
	return ret


def get_criterion(config):
	key = 'criterion'
	key = config[key].lower()
	if key == 'mse-sum':
		return nn.MSELoss(reduction='sum')
	elif key == 'cel-sum':
		return nn.CrossEntropyLoss(reduction='sum')
	else:
		return nn.CrossEntropyLoss(reduction='sum')

def get_optimizer(config, model):
	key = 'optimizer'
	key = config[key].lower()
	optimizer = None

	learning_rate = config['learning_rate']
	l2_lambda = config['l2_lambda']

	if key == 'adam':
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
	elif key == 'rmsprop':
		optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
	elif key == 'sgd':
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
	elif key == 'adadelta':
		optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
	return optimizer