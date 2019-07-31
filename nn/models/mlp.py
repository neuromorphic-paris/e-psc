import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class EMLP(torch.nn.Module):

	def __init__(self,c_in=3, c_out=[100,100],n_classes=10,kernel_size=500):
		super(EMLP, self).__init__()
		self.conv1 = nn.Conv1d(c_in,c_out[0],kernel_size)
		self.full1 = nn.Linear(c_out[0],c_out[1])
		self.full2 = nn.Linear(c_out[-1],n_classes)

	# def forward(self, x,return_layer_number=2):
	# 	out = []
	# 	for i in range(len(x)):
	# 		out.append(F.relu(self.conv1(x)).sum(-1))
	# 		out[-1] = self.full1(out[-1])
	# 		out[-1] = self.full2(out[-1])
	# 	return torch.cat(out)


	def forward(self, x,return_layer_number=3):
		if return_layer_number==0:
			return x
		out = F.relu(self.conv1(x)).mean(-1)
		if return_layer_number==1:
			return out
		out = self.full1(out)
		if return_layer_number==2:
			return out
		out = self.full2(out)
		if return_layer_number==3:
			return out

