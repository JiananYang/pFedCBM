

from collections import OrderedDict
from torch import Tensor
import torch.nn as nn
import torch
class ClassifierHead(nn.Module):

	def __init__(self,concept_bank,handbook,idx_to_class,classes,n_classes=5):
		super(ClassifierHead,self).__init__()
		self.concept_bank = concept_bank
		self.n_classes = n_classes
		
		self.norm = concept_bank.norms
		self.cavs = concept_bank.vectors
		self.intercepts = concept_bank.intercepts
		self.n_concepts = self.cavs.shape[0]

		self.handbook = handbook
		self.idx_to_class = idx_to_class if idx_to_class else {i: i for i in range(self.n_classes)}
		self.names = concept_bank.concept_names.copy()
		self.classes = classes

		self.classifier = nn.Linear(self.n_concepts,self.n_classes)


		

	def compute_dist(self,emb):
		margins = (torch.matmul(self.cavs,emb.T) + self.intercepts) / self.norm
		return margins.T

	def forward(self,emb):
		x = self.compute_dist(emb)
		out = self.classifier(x)
		return out
	
	def predict(self,emb):
		prediction = self.classifier(emb)
		_,label = torch.max(prediction,dim=1)
		return label

	def forward_projs(self, projs):
		# print (projs.shape)
		return self.classifier(projs)
	
	def trainable_params(self):
		return self.classifier.parameters()
	
	def classifier_weights(self):
		return self.classifier.weight
	
	def set_weights(self, weights, bias):
		self.classifier.weight.data = torch.tensor(weights).to(self.classifier.weight.device)
		self.classifier.bias.data = torch.tensor(bias).to(self.classifier.weight.device)
		return 1
	
	def state_dict(self):
		return self.classifier.state_dict()

	def load_state_dict(self, state_dict: OrderedDict[str, Tensor]):
		self.classifier.load_state_dict(state_dict)
		return 1

	def analyze_classifier(self,k=5,print_lows=False):
		weights = self.classifier.weight.clone().detach()
		output = []

		# if len(self.idx_to_class) == 2:
		# 	weights = [weights.squeeze(), weights.squeeze()]
		
		for idx, cls in self.idx_to_class.items():
			cls_weights = weights[idx]
			topk_vals, topk_indices = torch.topk(cls_weights, k=k)
			topk_indices = topk_indices.detach().cpu().numpy()
			topk_concepts = [self.names[j] for j in topk_indices]
			analysis_str = [f"Class : {cls}"]
			for j, c in enumerate(topk_concepts):
				analysis_str.append(f"\t {j+1} - {c}: {topk_vals[j]:.3f}")
			analysis_str = "\n".join(analysis_str)
			output.append(analysis_str)

			if print_lows:
				topk_vals, topk_indices = torch.topk(-cls_weights, k=k)
				topk_indices = topk_indices.detach().cpu().numpy()
				topk_concepts = [self.names[j] for j in topk_indices]
				analysis_str = [f"Class : {cls}"]
				for j, c in enumerate(topk_concepts):
					analysis_str.append(f"\t {j+1} - {c}: {-topk_vals[j]:.3f}")
				analysis_str = "\n".join(analysis_str)
				output.append(analysis_str)

		analysis = "\n".join(output)
		return analysis
