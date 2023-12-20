

import copy
import torch

def fedAvg(w):
	"""
	w_avg: dicts, keys are 
	"""

	w_avg = copy.deepcopy(w[0])
	for k in w_avg.keys():
		for i in range(1,len(w)):
			w_avg[k] += w[i][k]
		w_avg[k] = torch.div(w_avg[k],len(w))
	return w_avg