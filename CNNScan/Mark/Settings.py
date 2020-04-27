import CNNScan.Settings# Create sensible defaults for all the settings in the network.
def generate_default_settings():
	ret = CNNScan.Settings.generate_default_settings()
	ret['nlo']					= "lrelu"
	# Size of mark images, mark outputs
	ret['im_size']				= (4, 128, 128)

	# How many generated items should be mixed in with each batch?
	ret['generated_count']		= 10
	# How many random values should be used to seed the generator?
	ret['gen_seed_len']			= 10
	# How many embedding entires should be used to encode class labels?
	ret['gen_embed_size']		= 5
	ret['gen_full_layers']		= [200, 100, 100]

	# Discriminator configuraiton
	ret['disc_full_layers']		= [200, 100, 100]

	return ret