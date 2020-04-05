import math

import torch
import pylab as plt

# Our classes.
import Election as Election

# Determine if cuda is available on the machine
def cuda(arr, config):
    if config['cuda']:
        return arr.cuda()
    return arr

# Determine if a number is a power of 2 or not and the number is non-zero.
def is_power2(number):
	return number > 0 and math.ceil(math.log(number, 2)) == math.floor(math.log(number, 2)) 

# Return the next power of two larger than number, and the number of indices needed padding above and below the number.
def pad_nearest_pow2(number):
	next_pow2 = number
	pad_first, pad_second = 0,0
	if not is_power2(number):
		next_pow2 = 2**math.ceil(math.log(number, 2))
		needed_padding = next_pow2 - number
		pad_first = needed_padding // 2
		pad_second = needed_padding - pad_first
	return (next_pow2, pad_first, pad_second)
# Convert images to tensors, and apply normalization if necessary
def image_to_tensor(image):
	#TODO: apply image normalization.
	return torch.from_numpy(image)
	
# Visualize marked ballots.
def show_ballot(marked: Election.MarkedContest):
	plt.imshow(marked.image, cmap='gray', interpolation='nearest')
	plt.show()
