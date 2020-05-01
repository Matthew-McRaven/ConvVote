import CNNScan.Settings# Create sensible defaults for all the settings in the network.
def generate_default_settings():
	ret = CNNScan.Settings.generate_default_settings()
	ret['nlo']					= "lrelu"
	ret['learning_rate'] 		= 0.0001
	# Size of mark images, mark outputs
	ret['im_size']				= (2, 128, 128)
	ret['criterion'] 			= 'bce-sum'
	# How many random values should be used to seed the generator?
	ret['gen_seed_len']			= 100
	# How many embedding entires should be used to encode class labels?
	ret['gen_embed_size']		= 10
	ret['gen_full_layers']		= [256*15]
	ret['gen_conv_layers'] 	= [
		# Make sure the kernel size is SMALLER than the feature being recognized.
		CNNScan.Settings.conv_def(4, 4, 1, 0, 1, False),
	]

	# Discriminator configuraiton
	ret['disc_full_layers']		= [400, 200]

	return ret