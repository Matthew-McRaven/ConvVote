import math

import torch
import pylab as plt

# Our classes.
from CNNScan.Ballot import BallotDefinitions, MarkedBallots

# Determine if cuda is available on the machine
def cuda(arr, config):
    if config['cuda']:
        return arr.cuda()
    return arr

# Determine the size of a dimension after applying a pool / convolutional layer.
def resize_convolution(x, kernel_size, dilation, stride, padding):
    x = int(1 + (x + 2*padding - dilation * (kernel_size - 1) - 1)/stride)
    return x

# Determine the size of a dimension after applying a transposed convolution layer.
def resize_transpose_convolution(x, kernel_size, dilation, stride, padding, output_padding):
	t1 = (x-1)*stride
	t2 = 2*padding
	t3 = dilation*(kernel_size-1)
	t4 = output_padding
	return t1 - t2 + t3 + t4 + 1

	
# Determine if a number is a power of 2 or not and the number is non-zero.
def is_power2(number):
	return number > 0 and math.ceil(math.log(number, 2)) == math.floor(math.log(number, 2)) 

# Return the next power of two larger than number, and the number of indices needed padding above and below the number.
def pad_nearest_pow2(number, at_least_this=1):
	next_pow2 = number
	pad_first, pad_second = 0,0
	if not is_power2(number) or number < at_least_this:
		next_pow2 = 2**math.ceil(math.log(number, 2))
		if next_pow2 < at_least_this:
			next_pow2 = at_least_this
		needed_padding = next_pow2 - number
		pad_first = needed_padding // 2
		pad_second = needed_padding - pad_first
	return (next_pow2, pad_first, pad_second)
	
# Convert images to tensors, and apply normalization if necessary
def image_to_tensor(image):
	#TODO: apply image normalization.
	return torch.from_numpy(image)

# Visualize marked ballots.
def show_ballot(ballot:BallotDefinitions.Ballot, marked:MarkedBallots.MarkedBallot):
	count = len(marked.marked_contest)
	fig = plt.figure()
	for i, contest in enumerate(marked.marked_contest):
		ax = fig.add_subplot( math.ceil(count/5),5, i+1)
		ax.set_title(f'Contest {contest.index}')
		ax.set_xlabel(f'Vote for {contest.actual_vote_index}. Recorded as {contest.computed_vote_index}')
		ax.imshow(contest.image, interpolation='nearest')
	plt.show()

def labels_to_vec(labels, length):
	ret = [0]*length
	for label in labels:
		 ret[label] = 1
	return ret
