
import os
import torch
import shutil
import re
import pickle
from helpers import makedir
import numpy as np
import torch.nn.functional as F
from utils.options import args_parser,linear_probe_parser
from utils.sampling import CUB_iid,CUB_non_iid 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from log import create_logger
from preprocess import mean, std
from concepts import ConceptBank
# from datasets import get_dataset
from models import get_model
from datasets.param import backbone_name,dataset_name,out_dir,device,seed,num_workers,concept_batch_size,n_samples_concept,C
from models import PosthocLinearCBM
from training_tools import load_or_compute_projections
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from FLClassifier import *
from FL import fedAvg

class Client:
	def __init__(self,id,train_loader,test_loader,val_loader,backbone,classifierHead,log,args):
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.val_loader = val_loader
		self.backbone = backbone
		self.classifierHead = classifierHead
		self.classifierHead.to(args.device)
		self.id = id
		self.log = log
		self.args = args
		self.loss = nn.CrossEntropyLoss()
		self.device = torch.device('cuda:0')
		self.optim = torch.optim.SGD(self.classifierHead.parameters(),lr=0.1)
		self.classifier = SGDClassifier(random_state=args.seed, loss="log_loss",
							   alpha=args.lam, l1_ratio=args.alpha, verbose=0,
							   penalty="elasticnet", max_iter=10000)
		
	def train(self,weights):
		# self.classifier.fit(self.X_train,self.y_train)
		# print (self.train_loader.dataset[0])
		self.log("Training client:{}".format(self.id))
		print (weights.keys())
		self.classifierHead.load_state_dict(weights)
		num_epoch = 1000
		train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls=load_or_compute_projections(args, self.backbone, self.classifierHead, self.train_loader, self.test_loader)
		# print (train_lbls[0])
		# train_lbls = F.one_hot(train_lbls,200)
		train_lbls = train_lbls.to(self.device)
		test_lbls = test_lbls.to(self.device)
		# print (train_lbls.get_device())
		losses = []
		accs = []
		for epoch in range(num_epoch):
			y_pred = self.classifierHead.forward_projs(train_projs)

			y_pred = y_pred.to(self.device)
			# print (y_pred.get_device())
			# pred_label = torch.argmax(y_pred,dim=1).float()

			l = self.loss(y_pred,train_lbls)

			acc = self.score(test_projs,test_lbls)
			accs.append(acc)
			# print (acc)
			losses.append(l)
			# print (l)
			self.optim.zero_grad()
			l.backward()
			self.optim.step()
			if epoch % 100 == 0:
				print (acc)
		self.log("loss:{}".format(losses[-1]))
		
		# train_lbls = train_lbls.to(torch.float32)
		train_predictions = self.classifierHead.predict(train_projs)
		train_predictions = train_predictions.to(self.device)
		train_accuracy = torch.mean((train_lbls == train_predictions).to(torch.float32)) * 100.
		test_prediction = self.classifierHead.predict(test_projs)
		test_prediction = test_prediction.to(self.device)
		test_accuracy = torch.mean((test_lbls == test_prediction).to(torch.float32)) * 100.

		cls_acc = {"train": {}, "test": {}}
		for lbl in torch.unique(train_lbls):
			test_lbl_mask = test_lbls == lbl
			train_lbl_mask = train_lbls == lbl
			cls_acc["test"][lbl] = torch.mean((test_lbls[test_lbl_mask] == test_prediction[test_lbl_mask]).to(torch.float32))
			cls_acc["train"][lbl] = torch.mean(
				(train_lbls[train_lbl_mask] == train_predictions[train_lbl_mask]).to(torch.float32))
			log(f"{lbl}: {cls_acc['test'][lbl]}")

		run_info = {"train_acc": train_accuracy, "test_acc": test_accuracy,
					"cls_acc": cls_acc,
					}
		self.log("train_acc:{}".format(train_accuracy))
		self.log("test_acc:{}".format(test_accuracy))

		topkconcepts = self.classifierHead.analyze_classifier(k=5)
		self.log(topkconcepts)
		# run_info, weights, bias = self.run_linear_probe(args, (train_projs, train_lbls), (test_projs, test_lbls))
		# return run_info,weights,bias
		return accs,self.classifierHead.classifier.state_dict()
	def predict(self,X):
		return torch.argmax(self.classifierHead.forward_projs(X),dim=1)

	def score(self,X,y):
		train_predictions = self.predict(X)
		# print (train_predictions,y)
		train_accuracy = torch.mean((y == train_predictions).float())
		return train_accuracy.item()


	def run_linear_probe(self,args, train_data, test_data):
		log ("Start training for client:{}".format(self.id))
		train_features, train_labels = train_data
		test_features, test_labels = test_data
		
		# We converged to using SGDClassifier. 
		# It's fine to use other modules here, this seemed like the most pedagogical option.
		# We experimented with torch modules etc., and results are mostly parallel.
		classifier = SGDClassifier(random_state=args.seed, loss="log_loss",
								alpha=args.lam, l1_ratio=args.alpha, verbose=0,
								penalty="elasticnet", max_iter=10000)
		
		classifier.fit(train_features, train_labels)

		train_predictions = classifier.predict(train_features)
		train_accuracy = np.mean((train_labels == train_predictions).astype(float)) * 100.
		predictions = classifier.predict(test_features)
		test_accuracy = np.mean((test_labels == predictions).astype(float)) * 100.

		# Compute class-level accuracies. Can later be used to understand what classes are lacking some concepts.
		cls_acc = {"train": {}, "test": {}}
		for lbl in np.unique(train_labels):
			test_lbl_mask = test_labels == lbl
			train_lbl_mask = train_labels == lbl
			cls_acc["test"][lbl] = np.mean((test_labels[test_lbl_mask] == predictions[test_lbl_mask]).astype(float))
			cls_acc["train"][lbl] = np.mean(
				(train_labels[train_lbl_mask] == train_predictions[train_lbl_mask]).astype(float))
			log(f"{lbl}: {cls_acc['test'][lbl]}")

		run_info = {"train_acc": train_accuracy, "test_acc": test_accuracy,
					"cls_acc": cls_acc,
					}
		log("train_acc:{}".format(train_accuracy))
		log("test_acc:{}".format(test_accuracy))

		# If it's a binary task, we compute auc
		if test_labels.max() == 1:
			run_info["test_auc"] = roc_auc_score(test_labels, classifier.decision_function(test_features))
			run_info["train_auc"] = roc_auc_score(train_labels, classifier.decision_function(train_features))

		
		return run_info, classifier.coef_, classifier.intercept_


def get_handbook(dataset_name):
	handbook = open(os.path.join('data/concept_handbook',dataset_name+'.txt'),'r')
	concepts = []
	for concept in handbook.readlines():
		concepts.append(concept.strip().split(' ')[1])
	return concepts

class FL:
	def __init__(self,args,num_classes,n_clients=5):
		self.n_clients = n_clients
		self.clients = []
		self.args = args
		self.N = 0
		self.dataset_name = args.dataset_name
		self.num_classes = num_classes

	def create_clients(self,concept_bank,backbone,dict_user,idx_to_class,classes,log):

		for idx,lis in dict_user.items():
			handbook = get_handbook(self.args.concept_dataset_name)
			classifierHead = ClassifierHead(concept_bank,handbook,idx_to_class,classes,self.num_classes)
			client = Client(idx,lis[0],lis[1],lis[2],backbone,classifierHead,log,self.args)
			self.clients.append(client)

			

	def train(self):
		accs = []
		w_glob = self.clients[0].classifierHead.state_dict()
		for epoch in range(self.args.num_communications):
			weights = {}
			for i in range(self.n_clients):
				id = self.clients[i].id
				acc,weight = self.clients[i].train(w_glob)
				accs.append(copy.deepcopy(acc))
				weights[id] = copy.deepcopy(weight)
				# run_info,coef,intercept_ = self.clients[i].accuracy()
			w_glob = fedAvg(weights)
			
			# weights = copy.deepcopy(w_glob)
		return accs,w_glob
	
	def fedavg(self):
		accs = []
		w_glob = self.clients[0].classifierHead.state_dict()
		for epoch in range(self.args.num_communications):
			weights = {}
			for i in range(self.n_clients):
				id = self.clients[i].id
				acc,weight = self.clients[i].train(copy.deepcopy(w_glob))
				accs.append(copy.deepcopy(acc))
				weights[id] = copy.deepcopy(weight)
				# run_info,coef,intercept_ = self.clients[i].accuracy()
			w_glob = fedAvg(weights)
			
			# weights = copy.deepcopy(w_glob)
		return accs,w_glob
	
	# def fedprox(self):




if __name__ == "__main__":
	 

	#parse arguments
	args = linear_probe_parser()
	args.device = torch.device('cuda:{}'.format(args.gpu))
	from settings import base_architecture, img_size, prototype_shape, num_classes, \
						prototype_activation_function, add_on_layers_type, experiment_run

	base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

	model_dir = 'saved_models/' + '{}FL{}'.format(args.sampling,args.dataset_name)
	makedir(model_dir)
	# shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
	# shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
	# shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
	# shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
	# shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)
	log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train_classifier.log'))

	#we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
	
	#model construction

	#optimizer
	# from settings import joint_optimizer_lrs, joint_lr_step_size
	# joint_optimizer_specs = \
	# [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
	# {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
	# {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
	# ]
	# joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
	# joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)


	# number of training epochs, number of warm epochs, push start epoch, push epochs
	from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs
	import numpy as np
	import copy
	# from FL import fedAvg

	
	#Concept bank training
	# from concept_learner_fl import concept_learner

	#Linear probe training
	# concept_bank_name = "FLderm7pt/derm7pt_resnet18_cifar100_0.01_50.pkl"
	# backbone_name = "ham10000_inception"
	# concept_bank_name = "FLbroden/broden_resnet50_cifar_0.01_50.pkl"
	# backbone_name = "resnet50_cifar"
	concept_bank_path = f"{args.sampling}FL{args.concept_dataset_name}/{args.concept_dataset_name}_{args.backbone_name}_{args.C[0]}_{args.n_samples_concept}.pkl"
	# print (concept_bank_name)
	# concept_bank_name = "{}FL{}/".format(args.sampling,args.concept_dataset_name)
	# if args.dataset_name == 'cub':
	# 	concept_bank_name = "FLcub/cub_resnet18_cifar100_0.1_50.pkl"
	# 	backbone_name = "resnet18_cub"
	# elif dataset_name == 'cifar10' or dataset_name == 'cifar100':
	# 	concept_bank_name = "FLbroden/broden_resnet18_cub_0.1_50.pkl"
	# 	backbone_name = "resnet50_cifar"
	# elif dataset_name == 'derm7pt' or dataset_name == 'ham10000':
	# 	concept_bank_name = "FLderm7pt/derm7pt_resnet18_cub_0.1_50.pkl"
	# 	backbone_name = "ham10000_inception"
	all_concepts = pickle.load(open(os.path.join('saved_models',concept_bank_path), 'rb'))
	all_concept_names = list(all_concepts.keys())
	print(f"Bank path: {concept_bank_path}. {len(all_concept_names)} concepts will be used.")
	concept_bank = ConceptBank(all_concepts, args.device)

	model,backbone, preprocess = get_model(args)
	backbone = backbone.to(args.device)
	backbone.eval()

	fl = FL(args,args.num_classes)
	from datasets import load_data
	dataset_dict_user,idx_to_class,classes = load_data(args,preprocess=preprocess)

	fl.create_clients(concept_bank,backbone,dataset_dict_user,idx_to_class,classes,log)

	# log('training set size: {0}'.format(len(train_loader.dataset)))
	# log('test set size: {0}'.format(len(test_loader.dataset)))
	# log('batch size: {0}'.format(train_batch_size))
	accs,w_glob = fl.fedavg()
	torch.save(w_glob,os.path.join(model_dir,"fedavg_{}.pt".format(args.dataset_name)))

	with open(os.path.join(model_dir,"{}_{}_concept_predict.pkl".format(args.sampling,args.dataset_name)),'wb') as f:
		pickle.dump(accs,f)

	# train_loader, test_loader, idx_to_class, classes = get_dataset(args,preprocess)
	# fl.create_clients(train_loader,test_loader)
	# conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
	# num_classes = len(classes)
	
	# # Initialize the PCBM module.
	# posthoc_layer = PosthocLinearCBM(concept_bank, backbone_name=args.backbone_name, idx_to_class=idx_to_class, n_classes=num_classes)
	# posthoc_layer = posthoc_layer.to(args.device)

	# # We compute the projections and save to the output directory. This is to save time in tuning hparams / analyzing projections.
	# train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader)
	# run_info, weights, bias = run_linear_probe(args, (train_projs, train_lbls), (test_projs, test_lbls))
	
	# # Convert from the SGDClassifier module to PCBM module.
	# posthoc_layer.set_weights(weights=weights, bias=bias)

	

	# model_path = os.path.join(args.out_dir,
	# 						  f"pcbm_{args.dataset}__{args.backbone_name}__{conceptbank_source}__lam{args.lam}__alpha{args.alpha}__seed{args.seed}.ckpt")
	# torch.save(posthoc_layer, model_path)

	# run_info_file = model_path.replace("pcbm", "run_info-pcbm")
	# run_info_file = run_info_file.replace(".ckpt", ".pkl")
	# # run_info_file = os.path.join(args.out_dir, run_info_file)
	
	# with open(run_info_file, "wb") as f:
	# 	pickle.dump(run_info, f)

	
	# if num_classes > 1:
	# 	# Prints the Top-5 Concept Weigths for each class.
	# 	print(posthoc_layer.analyze_classifier(k=5))

	# print(f"Model saved to : {model_path}")
	# print(run_info)

