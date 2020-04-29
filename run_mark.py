# Begin training neural net using default data / parameters
import torch
import torchvision
import numpy as np

from CNNScan.Mark import gan
import CNNScan.Mark.Settings

# Choose to use real Oregon data (on which the network performs poorly)
# Or choose randomly generate data, on which the network performs decently.
config = CNNScan.Mark.Settings.generate_default_settings()

transforms = torchvision.transforms.Compose([
											 #torchvision.transforms.Grayscale(),
 											 torchvision.transforms.ToTensor(),
											 #torchvision.transforms.Normalize((1,),(127.5,))
											 #torchvision.transforms.Lambda(lambda x: (x[0] + x[1] + x[2])/3)
											])

data = CNNScan.Mark.gan.get_marks_dataset(CNNScan.Mark, transforms)
loader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True)

disc_model = CNNScan.Mark.gan.MarkDiscriminator(config)
gen_model = CNNScan.Mark.gan.MarkGenerator(config, config['gen_seed_len'])

print(disc_model)
print(gen_model)

config['epochs'] = 50
CNNScan.Mark.gan.train_once(config, gen_model, disc_model, loader, loader)

count=4
images = CNNScan.Mark.gan.generate_images(gen_model, count, config, torch.tensor(count*[1]))
print(images.shape)
toImage= torchvision.transforms.Compose([
										 torchvision.transforms.Normalize((-1/127.5,),(1/127.5,)),
 									     torchvision.transforms.ToPILImage(mode='LA')
										])
for image in images:
	#print(image.shape)
	#image = (torchvision.transforms.Normalize((-1/127.5,),(1/127.5,))(image)).type(torch.ByteTensor)
	print(image)
	toImage(image).show()

