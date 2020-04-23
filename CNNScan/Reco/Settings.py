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
	ret['criterion'] 		= 'BCE-sum'
	# See Settings::get_criterion() for available criterion.
	ret['optimizer'] 		= 'adam'

	# Model hyperparameters
	ret['learning_rate'] 	= 0.0001
	ret['l2_lambda'] 		= 0.01
	ret['dropout'] 			= 0.1
	ret['epochs'] 			= 10
	ret['batch_size']		= 7

	# Enable / disable pooling in rescaler
	ret['rescale_pooling'] 		= False
	# Determine how to pad / cut the images
	ret['target_resolution']	= (128, 128)
	# Number of channels coming out of the rescaler.
	ret['target_channels']		= 4

	# Determine how the inner CNN / FC layers are built
	ret['recog_conv_nlo']		= "ReLu"
	ret['recog_copies']			= 5
	ret['recog_conv_layers'] 	= [
		# Make sure the kernel size is SMALLER than the feature being recognized.
		conv_def(4, 4, 1, 0, 1, False),
		#conv_def(4, 16, 1, 0, 1, False),
		pool_def(1, 1, 0, 1, True, 'avg')
		#conv_def(6, 8, 2, 1, 2, False),
		#conv_def(6, 8, 2, 1, 2, False),
		#pool_def(4, 1, 0, 1, True, 'avg')
		]
	ret['unique_outputs']		= True
	ret['recog_full_nlo']		= "ReLu"
	ret['recog_embed']			= 5
	ret['recog_full_layers']	= [100]
	#ret['recog_full_layers']	= [1600, 1000, 800, 600]

	# How many values should come out of the ImageRecognitionCore class?

	return ret

# Class containing overlapping parameters been convolutional, pooling layers
class conpool_core:
	def __init__(self, kernel, stride, padding, dilation, non_linear_after):
		self.kernel = kernel
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.non_linear_after = non_linear_after

# Class describing a 2D convolutional layer
class conv_def(conpool_core):
	def __init__(self, kernel, out_channels, stride=1, padding=0, dilation=1, non_linear_after=True):
		super(conv_def, self).__init__(kernel, stride, padding, dilation, non_linear_after)
		self.out_channels = out_channels

# Class describing a 2D pooling layer.
class pool_def(conpool_core):
	def __init__(self, kernel, stride=None, padding=0, dilation=1, non_linear_after=False, pool_type='avg'):
		super(pool_def, self).__init__(kernel, stride, padding, dilation, non_linear_after)
		if stride is None:
			self.stride = self.kernel
		self.pool_type = pool_type

def get_nonlinear(non_linear_name):
	if non_linear_name.lower() == "relu":
		return nn.ReLU()
	else:
		raise NotImplementedError(f"{non_linear_name} is not an implemented non-linear operator")

def get_criterion(config):
	key = 'criterion'
	key = config[key].lower()
	if key == 'mse-sum':
		return nn.MSELoss(reduction='sum')
	elif key == 'cel-sum':
		return nn.CrossEntropyLoss(reduction='sum')
	elif key == 'bcelog-sum':
		return nn.BCEWithLogitsLoss(reduction='sum')
	elif key == 'bce-sum':
		return nn.BCELoss(reduction='sum')
	else:
		raise NotImplementedError(f"{key} is not an implemented loss criterion function.")

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
	else:
		raise NotImplementedError(f"Optimize '{key}' is not yet implemented.'")
	return optimizer
