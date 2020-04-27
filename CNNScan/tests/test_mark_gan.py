import unittest
# Begin training neural net using default data / parameters
import torch
import torchvision
import numpy as np

from CNNScan.Mark import gan
import CNNScan.Mark.Settings






class TestMarkGan(unittest.TestCase):
		

	# Check that the (discriminator, generator) pair runs without crashing.
	def test_gan_pair(self):
		config = CNNScan.Mark.Settings.generate_default_settings()
		config['epochs'] = 1

		transforms = torchvision.transforms.Compose([
 											 torchvision.transforms.ToTensor()
											])
											
		data = CNNScan.Mark.gan.get_marks_dataset(CNNScan.Mark, transforms)
		loader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True)

		# Create discriminator, generators for GAN.
		disc_model = CNNScan.Mark.gan.MarkDiscriminator(config)
		gen_model = CNNScan.Mark.gan.MarkGenerator(config, config['gen_seed_len'])

		if False:
			print(disc_model)
			print(gen_model)

		CNNScan.Mark.gan.train_once(config, gen_model, disc_model, loader, loader)

if __name__ == '__main__':
    unittest.main()