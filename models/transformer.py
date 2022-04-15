from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad



##-----------------------------------------------------------------------------
# Class for Transformer. Subclasses PyTorch's own "nn" module
#
# Computes a KxK affine transform from the input data to transform inputs
# to a "canonical view"
##
class transformer(nn.Module):

	def __init__(self, points_num, K=3):
		# Call the super constructor
		super(transformer, self).__init__()

		# Number of dimensions of the data
		self.K = K

		# Size of input
		self.N = points_num

		# Initialize identity matrix on the GPU (do this here so it only 
		# happens once)
		self.identity = grad.Variable(
			torch.eye(self.K).double().view(-1).cuda())

		# First embedding block
		self.block1 =nn.Sequential(
			nn.Conv1d(K, 64, 1),
			nn.BatchNorm1d(64),
			nn.ReLU())

		# Second embedding block
		self.block2 =nn.Sequential(
			nn.Conv1d(64, 128, 1),
			nn.BatchNorm1d(128),
			nn.ReLU())

		# Third embedding block
		self.block3 =nn.Sequential(
			nn.Conv1d(128, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU())

		# Multilayer perceptron
		self.mlp = nn.Sequential(
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Linear(128, K)
		)


	# Take as input a B x K x N matrix of B batches of N points with K 
	# dimensions
	def forward(self, x):

		# Compute the feature extractions
		# Output should ultimately be B x 1024 x N
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)

		# Pool over the number of points
		# Output should be B x 1024 x 1 --> B x 1024 (after squeeze)
		x = F.max_pool1d(x, self.N).squeeze(2)
		
		# Run the pooled features through the multi-layer perceptron
		# Output should be B x K^2
		x = self.mlp(x)

		# # Add identity matrix to transform
		# # Output is still B x K^2 (broadcasting takes care of batch dimension)
		# x += self.identity

		# # Reshape the output into B x K x K affine transformation matrices
		# x = x.view(-1, self.K, self.K)

		return x

def test_transformer():
    # trans = self.stn(x)
    # x = x.transpose(2, 1)
    # x = torch.bmm(x, trans)
    # x = x.transpose(2, 1)
    bs, pn, d = 2, 8, 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.rand((bs, pn,d), dtype=torch.float32, device=device)
    
    tnet = transformer(points_num=pn).to(device=device)

    trans = tnet(x.transpose(1,2))
    print("\n------------------trans:--------------------------\n", trans)
    # x = torch.bmm(x, trans)
    print("xtrans size", trans.size())
    
    # coords = [ME.utils.sparse_quantize(coordinates=coord.detach().numpy(), quantization_size=0.01)
    #             for coord in x]
    return

if __name__ == "__main__":
    test_transformer()