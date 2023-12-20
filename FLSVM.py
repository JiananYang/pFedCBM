
import numpy as np
import copy
import torch
from PIL import Image
import pickle
import os
from helpers import makedir
# import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# from datasets.param import *
from log import create_logger
# from settings import dataset_name,out_dir,device,seed,num_workers,concept_batch_size,n_samples_concept,C,CUB_ATTRIBUTE_DIR
from torch.utils.data import Dataset, DataLoader
from datasets.concept_loaders import *
from models.model_zoo import get_model
from utils.options import concept_args_parser
class SVM:

	def __init__(self, X_train, y_train, X_test, y_test, c,val=True, val_type='k_fold', k=5, opt='mini_batch_GD', batch_size = 30, n_iters=100, learning_rate=0.001):
		self.lr = learning_rate
		self.lambda_param = c
		self.n_iters = n_iters

		self.X_train = X_train
		self.y_train = y_train

		self.X_test = X_test
		self.y_test = y_test

		self.val = val
		self.val_type=val_type
		self.k=k

		self.opt = opt
		self.batch_size = batch_size

		self.w = np.zeros(self.X_train.shape[1])
		self.b = 0

	def grad(self,x,y):

		if y * (np.dot(x, self.w) - self.b) >= 1:
			# dw = self.lr * (2 * self.lambda_param * self.w)
			dw = 0
			db = 0
		else:
			dw = self.lr * (-np.dot(x, y))
			db = self.lr * y

		return (dw,db)
	
	def localgrad(self,weight,bias):
		self.w = weight
		self.b = bias
		y_ = np.where(self.y_train <= 0, -1, 1)
		dw_avg = 0
		db_avg = 0
		for idx,x_i in enumerate(self.X_train):
			dw,db = self.grad(x_i,y_[idx])
			dw_avg += dw
			db_avg += db
		# dw_avg /= len(self.X_train)
		# db_avg /= len(self.X_train)
		return dw_avg,db_avg
		

	def loss(self):
		return np.mean([max(0, 1-x*y) for x, y in zip(np.where(np.concatenate(self.y_train,axis=None) <= 0, -1, 1), self.predict())])

	def stochastic_GD(self, X_train, y_train, X_val=None, y_val=None):
		n_samples, n_features = X_train.shape  
		y_ = np.where(y_train <= 0, -1, 1)
				
		# if self.w.size == 0 and self.b is None :
		# 	self.w = np.zeros(n_features)
		# 	self.b = 0

		w_best = np.zeros(n_features)
		b_best = 0

		acc_list = [] 
		for i in range(0,self.n_iters):
			for idx, x_i in enumerate(X_train):
				dw,db = self.grad(x_i,y_[idx])
				self.w -= dw
				self.b -= db

		# 	if i%10 == 0 and self.val:
		# 		approx_w = np.dot(X_val, self.w) - self.b
		# 		approx_w = np.sign(approx_w)
		# 		res_w = np.where(approx_w<0, 0, approx_w)

		# 		approx_w_best = np.dot(X_val, w_best) - b_best
		# 		approx_w_best = np.sign(approx_w_best)
		# 		res_w_best = np.where(approx_w_best<0, 0, approx_w_best)
					
		# 		if (accuracy_score(y_val, res_w_best) < accuracy_score(y_val, res_w)):
		# 			w_best = copy.deepcopy(self.w)
		# 			b_best = copy.deepcopy(self.b)
		# self.w = w_best
		# self.b = b_best


	def batch_GD(self, X_train, y_train, X_val=None, y_val=None):
		n_samples, n_features = X_train.shape  
		y_ = np.where(y_train <= 0, -1, 1)
			
		if self.w.size == 0 and self.b is None :
			self.w = np.zeros(n_features)
			self.b = 0

		w_best = np.zeros(n_features)
		b_best = 0

		acc_list = [] 
		for i in range(0,self.n_iters):
			dw_sum=0
			db_sum=0
			for idx, x_i in enumerate(X_train):
				dw,db = self.grad(x_i,y_[idx])
				dw_sum+=dw
				db_sum+=db
			self.w -= (dw_sum/n_samples)
			self.b -= (db_sum/n_samples)
		
		if i%10 == 0 and self.val:
			approx_w = np.dot(X_val, self.w) - self.b
			approx_w = np.sign(approx_w)
			res_w = np.where(approx_w<0, 0, approx_w)

			approx_w_best = np.dot(X_val, w_best) - b_best
			approx_w_best = np.sign(approx_w_best)
			res_w_best = np.where(approx_w_best<0, 0, approx_w_best)
				
			if (accuracy_score(y_val, res_w_best) < accuracy_score(y_val, res_w)):
				w_best = copy.deepcopy(self.w)
				b_best = copy.deepcopy(self.b)


	def mini_batch_GD(self, X_train, y_train, X_val=None, y_val=None):
		n_samples, n_features = X_train.shape  
		y_ = np.where(y_train <= 0, -1, 1)
			
		if self.w.size == 0 and self.b is None :
			self.w = np.zeros(n_features)
			self.b = 0

		w_best = np.zeros(n_features)
		b_best = 0

		acc_list = [] 

		# print(self.n_iters)
		
		for i in range(0,self.n_iters):
		# print(i)
			dw_sum=0.0
			db_sum=0.0
			s=0
			for idx, x_i in enumerate(X_train):
				dw,db = self.grad(x_i,y_[idx])
				dw_sum+=dw
				db_sum+=db
				s += 1
				if s%self.batch_size==0:
					self.w -= (dw_sum/self.batch_size)
					self.b -= (db_sum/self.batch_size)
		
		if i%10 == 0 and self.val:
			approx_w = np.dot(X_val, self.w) - self.b
			approx_w = np.sign(approx_w)
			res_w = np.where(approx_w<0, 0, approx_w)

			approx_w_best = np.dot(X_val, w_best) - b_best
			approx_w_best = np.sign(approx_w_best)
			res_w_best = np.where(approx_w_best<0, 0, approx_w_best)
				
			if (accuracy_score(y_val, res_w_best) < accuracy_score(y_val, res_w)):
				w_best = copy.deepcopy(self.w)
				b_best = copy.deepcopy(self.b)

	def predict(self,X_test):
		approx = np.dot(X_test, self.w) - self.b
		approx = np.sign(approx)
		return np.where(approx<0, 0, approx)

	def accuracy(self,X_test):
		return accuracy_score(self.y_test, self.predict(X_test))

class FLSVM:
	def __init__(self,C,n_clients=5):
		self.n_clients = n_clients
		self.clients = []
		self.N = 0 #total number of samples
		self.w = 0
		self.b = 0
		self.d = 0 #embedding space dim
		self.c = C
		

	def create_clients(self,X_train,y_train,X_test,y_test,args):
		#split data to all clients
		from utils.sampling import CUB_iid,CUB_non_iid
		# print (args.sampling)
		if args.sampling == 'iid':
			train_dict_users = CUB_iid(X_train,self.n_clients)
			test_dict_users = CUB_iid(X_test,self.n_clients)
		else:
			train_dict_users = CUB_non_iid(X_train,self.n_clients)
			test_dict_users = CUB_non_iid(X_test,self.n_clients)
		# print (X_train.shape,y_train.shape,X_train[0].shape)
		self.X_test = X_test
		self.y_test = y_test
		for idx,samples in train_dict_users.items():

			samples = np.array(list(samples))
			X_train_client = X_train[samples,:]
			y_train_client = y_train[samples]
			X_test_client = X_test[samples,:]
			y_test_client = y_test[samples]

			# print (X_train_client.shape,y_train_client.shape)
			client = SVM(X_train_client,y_train_client,X_test_client,y_test_client,self.c)
			self.clients.append(client)
		self.N = X_train.shape[0]
		self.d = X_train.shape[1]

	# def fine_tune(self):
	# 	backbone

	def fit(self):

		iters = 100
		self.w = np.zeros(self.d)
		self.b = 0
		accs = []
		for i in range(iters):
			dw = 0
			db = 0
			acc = 0
			for j in range(self.n_clients):
				dwk,dbk = self.clients[j].localgrad(self.w,self.b)
				Nk = self.clients[j].X_train.shape[0]
				dw += dwk * (Nk / self.N)
				db += dbk * (Nk / self.N)
				self.w -= dw
				self.b -= db
				acc += self.accuracy(self.X_test,self.y_test)
			acc /= self.n_clients
			accs.append(acc)
		# print (accs)
		return accs
	def predict(self,X_test):
		approx = np.dot(X_test, self.w) - self.b
		approx = np.sign(approx)
		return np.where(approx<0, 0, approx)

	def accuracy(self,X_test,y_test):
		return accuracy_score(y_test, self.predict(X_test))


N_ATTRIBUTES=312
class ResNetBottom(nn.Module):
	def __init__(self, original_model):
		super(ResNetBottom, self).__init__()
		self.features = nn.Sequential(*list(original_model.children())[:-1])
	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		return x



def get_concept_dicts(metadata):
	""""
	meta:[{'id':0,'label':21,'attribute_label':[binary classifier of 112 concepts]},{},...]
	"""
	from settings import CUB_DATA_DIR
	n_concepts = len(metadata[0]["attribute_label"])
	concept_info = {c: {1: [], 0: []} for c in range(n_concepts)}
	for im_data in metadata:
		for c, label in enumerate(im_data["attribute_label"]):
			# print(c)
			img_path = im_data["img_path"]
			# print (img_path)            
			idx = img_path.split('/').index('CUB_200_2011')
			img_path = '/'.join([CUB_DATA_DIR] + img_path.split('/')[idx+1:])
			# print (img_path)
			concept_info[c][label].append(img_path)
	return concept_info

class CUBConceptDataset:
	def __init__(self, images, transform=None):
		self.images = images
		self.transform = transform

	def __len__(self):
		# Return the length of the dataset
		return len(self.images)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_path = self.images[idx]
		image = Image.open(img_path).convert('RGB')
		if self.transform:
			image = self.transform(image)
		return image
	
def get_embeddings(loader, model, device="cuda"):
	"""
	Args:
		loader ([torch.utils.data.DataLoader]): Data loader returning only the images
		model ([nn.Module]): Backbone
		n_samples (int, optional): Number of samples to extract the activations
		device (str, optional): Device to use. Defaults to "cuda".

	Returns:
		np.array: Activations as a numpy array.
	"""
	activations = None
	for image in tqdm(loader):
		image = image.to(device)
		try:
			batch_act = model(image).detach().cpu().numpy()
		except:
			# Then it's a CLIP model. This is a really nasty soln, i should fix this.
			batch_act = model.encode_image(image).detach().cpu().numpy()
		if activations is None:
			activations = batch_act
		else:
			activations = np.concatenate([activations, batch_act], axis=0)
	# print (activations.shape)
	return activations

def get_cavs(X_train, y_train, X_val, y_val, C,args):
	"""Extract the concept activation vectors and the corresponding stats

	Args:
		X_train, y_train, X_val, y_val: activations (numpy arrays) to learn the concepts with.
		C: Regularizer for the SVM. 
	"""
	# svm = SVM(X_train,y_train,X_val,y_val,C)
	# svm = FLSVM(C)
	# svm.create_clients(X_train,y_train,X_val,y_val,args)
	# accs = svm.fit()
	svm = SVC(C=C,kernel='linear')
	accs = []
	# train_acc = svm.accuracy(X_train,y_train)
	# test_acc = svm.accuracy(X_val,y_val)
	# train_margin = ((np.dot(svm.w, X_train.T) + svm.b) / np.linalg.norm(svm.w)).T
	svm.fit(X_train,y_train)
	train_acc = svm.score(X_train, y_train)
	test_acc = svm.score(X_val, y_val)
	train_margin = ((np.dot(svm.coef_, X_train.T) + svm.intercept_) / np.linalg.norm(svm.coef_)).T
	margin_info = {"max": np.max(train_margin),
				   "min": np.min(train_margin),
				   "pos_mean":  np.nanmean(train_margin[train_margin > 0]),
				   "pos_std": np.nanstd(train_margin[train_margin > 0]),
				   "neg_mean": np.nanmean(train_margin[train_margin < 0]),
				   "neg_std": np.nanstd(train_margin[train_margin < 0]),
				   "q_90": np.quantile(train_margin, 0.9),
				   "q_10": np.quantile(train_margin, 0.1),
				   "pos_count": y_train.sum(),
				   "neg_count": (1-y_train).sum(),
				   }
	# concept_info = (svm.w, train_acc, test_acc, svm.b, margin_info)
	concept_info = (svm.coef_, train_acc, test_acc, svm.intercept_, margin_info)
	return concept_info,accs
def learn_concept_bank(pos_loader, neg_loader,backbone, n_samples, C, log,args,device="cuda"):
	"""Learning CAVs and related margin stats.
	Args:
		pos_loader (torch.utils.data.DataLoader): A PyTorch DataLoader yielding positive samples for each concept
		neg_loader (torch.utils.data.DataLoader): A PyTorch DataLoader yielding negative samples for each concept
		model_bottom (nn.Module): Mode
		n_samples (int): Number of positive samples to use while learning the concept.
		C (float): Regularization parameter for the SVM. Possibly multiple options.
		device (str, optional): Device to use while extracting activations. Defaults to "cuda".

	Returns:
		dict: Concept information, including the CAV and margin stats.
	"""
	print("Extracting Embeddings: ")
	pos_act = get_embeddings(pos_loader, backbone, device=device)
	neg_act = get_embeddings(neg_loader, backbone, device=device)
	
	X_train = np.concatenate([pos_act[:n_samples], neg_act[:n_samples]], axis=0)
	X_val = np.concatenate([pos_act[n_samples:], neg_act[n_samples:]], axis=0)
	y_train = np.concatenate([np.ones(pos_act[:n_samples].shape[0]), np.zeros(neg_act[:n_samples].shape[0])], axis=0)
	y_val = np.concatenate([np.ones(pos_act[n_samples:].shape[0]), np.zeros(neg_act[n_samples:].shape[0])], axis=0)
	concept_info = {}
	# coef
	for c in C:
		concept_info[c],accs = get_cavs(X_train, y_train, X_val, y_val, c,args)
		log('c:{}'.format(c))
		log('train accuracy:{}'.format(concept_info[c][1]))
		log('test accuracy:{}'.format(concept_info[c][2]))
	# print (accs)
	return concept_info, accs, pos_act, neg_act



if __name__ == "__main__":
	
	args = concept_args_parser()
	args.device = torch.device('cuda:{}'.format(args.gpu))
	original_model,backbone,preprocess = get_model(args)
	print (type(backbone))
	backbone = backbone.to(device)
	backbone = backbone.eval()
	print (args.dataset_name)
	model_dir = out_dir + '/' + "{}FL".format(args.sampling) + args.dataset_name
	print (model_dir)
	makedir(model_dir)
	log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train_concept_bank.log'))

	concept_libs = {c: {} for c in C}
	# Get the positive and negative loaders for each concept.
	#cub concept_loaders
	# concept_loaders = cub_concept_loaders(preprocess, n_samples=n_samples_concept, batch_size=concept_batch_size, 
	# 									num_workers=num_workers, seed=seed)
	
	#derm7pt loaders

	concept_loaders = get_concept_loaders(args.dataset_name,preprocess, n_samples=args.n_samples_concept, batch_size=args.batch_size, 
										num_workers=args.num_workers, seed=args.seed)
	# concept_loaders = derm7pt_concept_loaders()
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	# concept_loaders = {}
	# print (len(concept_info),concept_info.keys())
	# c_test_label = 1
	# c_data = concept_info[c_test_label]
	
	#fine tune resnet
	

	# train_accuracy = [0 for _ in range(len(concept_loaders.keys()))]
	# test_accuracy = [0 for _ in range(len(concept_loaders.keys()))]
	accs = {}
	embeddings = {}
	for concept_name, loaders in tqdm(concept_loaders.items()):
		log('Concept:{}'.format(concept_name))
		pos_loader, neg_loader = loaders['pos'], loaders['neg']
		# Get CAV for each concept using positive/negative image split
		cav_info,acc,pos_act,neg_act = learn_concept_bank(pos_loader, neg_loader,backbone, args.n_samples_concept, args.C,log,args, device="cuda")
		accs[concept_name] = acc
		embeddings[concept_name] = {'pos':copy.deepcopy(pos_act),'neg':copy.deepcopy(neg_act)}
		for c in C:
			concept_libs[c][concept_name] = cav_info[c]
			# print (cav_info[c][-1])
	with open(os.path.join(model_dir,"embeddings.pkl"),'wb') as f:
		pickle.dump(embeddings,f)
	# torch.save(os.path.join(model_dir,"embeddings."),embeddings)
	with open("saved_models/{}_{}_concept.pkl".format(args.dataset_name,args.sampling),'wb') as f:
		pickle.dump(accs,f)
	for key in concept_libs.keys():
		lib_path = os.path.join(model_dir, f"{args.dataset_name}_{args.backbone_name}_{key}_{args.n_samples_concept}.pkl")
		with open(lib_path, "wb") as f:
			pickle.dump(concept_libs[key], f)
		print(f"Saved to: {lib_path}")        
	
	# 	total_concepts = len(concept_libs[key].keys())
	# 	print(f"File: {lib_path}, Total: {total_concepts}")
	# svm = SVM(X_train,y_train,X_val,y_val)
	# svm.stochastic_GD(X_train,y_train)
	# print (svm.accuracy(X_train))
	# print (svm.accuracy(X_val))

	# svm = SVC(C=0.1, kernel="linear")
	# svm.fit(X_train, y_train)
	# train_acc = svm.score(X_train, y_train)
	# test_acc = svm.score(X_val, y_val)
	# print (train_acc,test_acc)
	# for img in pos_loader:
	# 	print (img.shape)