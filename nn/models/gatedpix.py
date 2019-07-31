import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class EMLP(torch.nn.Module):

	def __init__(self,c_in=3, c_out=[100,100,50],n_classes=10,kernel_size=[500,4]):
		super(EMLP, self).__init__()
		self.convw = nn.Conv1d(1,c_out[0],kernel_size[0])
		self.convh = nn.Conv1d(1,c_out[0],kernel_size[0])
		# self.convt = nn.Conv1d(1,c_out[0],kernel_size[0])
		self.conv2 = nn.Conv2d(1,c_out[1],kernel_size[1])
		_padding = 0
		_dilation = 1
		_stride = 1
		self.out2size = ((c_out[0] - 2*_padding - _dilation*(kernel_size[1]-1)-1)/_stride)+1
		self.full1 = nn.Linear(c_out[1]*self.out2size**2,c_out[2])
		self.full2 = nn.Linear(c_out[-1],n_classes)


	def forward(self, x,return_layer_number=4):
		if return_layer_number==0:
			return x
		ow = F.relu(self.conv1(x[:,0]))
		oh = F.relu(self.conv1(x[:,1]))
		out = (ow[:,:,None,:]*oh[:,None,:,:]).mean(-1)
		if return_layer_number==1:
			return out
		out = self.conv2(out[:,None,:,:]).sum(-1).sum(-1)
		if return_layer_number==2:
			return out
		out = self.full1(out)
		if return_layer_number==3:
			return out
		out = self.full2(out)
		if return_layer_number==4:
			return out

