import CNNScan.Settings# Create sensible defaults for all the settings in the network.
def generate_default_settings():
	ret = CNNScan.Settings.generate_default_settings()
	# Enable / disable pooling in rescaler
	ret['rescale_pooling'] 		= False
	# Determine how to pad / cut the images
	ret['target_resolution']	= (128, 128)
	# Number of channels coming out of the rescaler.
	ret['target_channels']		= 4

	# Determine how the inner CNN / FC layers are built
	ret['recog_conv_nlo']		= "ReLu"
	ret['recog_conv_layers'] 	= [
		CNNScan.Settings.conv_def(4, 4, 1, 0, 1, False),
		CNNScan.Settings.pool_def(1, 1, 0, 1, True, 'avg')
		]
	ret['unique_outputs']		= True
	ret['recog_full_nlo']		= "ReLu"
	ret['recog_embed']			= 10
	ret['recog_full_layers']	= [400, 400]
	#ret['recog_full_layers']	= [1600, 1000, 800, 600]

	return ret