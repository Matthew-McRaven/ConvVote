import torch
import torch.nn as nn

# Create sensible defaults for settings shared between all networks
def generate_default_settings():
	ret = {}
	# Global configuration
	# Determine globally if CUDA should be enabled or disabled.
	ret['cuda'] 			= torch.cuda.is_available()

	# See Settings::get_criterion() for available criterion.
	ret['criterion'] 		= 'mse-sum'
	# See Settings::get_criterion() for available criterion.
	ret['optimizer'] 		= 'adam'

	# Model hyperparameters
	ret['learning_rate'] 	= 0.0001
	ret['l2_lambda'] 		= 0.01
	ret['dropout'] 			= 0.1
	ret['epochs'] 			= 10
	ret['batch_size']		= 20

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
	if non_linear_name.lower() == "lrelu":
		return nn.LeakyReLU()
	else:
		raise NotImplementedError(f"{non_linear_name} is not an implemented non-linear operator")

def get_criterion(config, criterion_string='criterion'):
	key = config[criterion_string].lower()
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
