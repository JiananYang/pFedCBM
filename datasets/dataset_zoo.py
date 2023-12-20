
import torch
from PIL import Image
import pickle
import os
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from tqdm import tqdm
import copy
from sklearn.model_selection import train_test_split
from .derm7data import DermDataset
from .param import *
# from .param import backbone_name,dataset_name,out_dir,device,seed,num_workers,concept_batch_size,n_samples_concept,C
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

N_ATTRIBUTES = 312
class CUBDataset(Dataset):
	"""
	Returns a compatible Torch Dataset object customized for the CUB dataset
	"""

	def __init__(self, dataset,image_dir, num_classes, transform=None, pkl_itself=None):
		"""
		Arguments:
		pkl_file_paths: list of full path to all the pkl data
		use_attr: whether to load the attributes (e.g. False for simple finetune)
		no_img: whether to load the images (e.g. False for A -> Y model)
		uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
		image_dir: default = 'images'. Will be append to the parent dir
		n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
		transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
		"""
		self.data = dataset
		# self.is_train = any(["train" in path for path in pkl_file_paths])
		# if not self.is_train:
		#     assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])        
		# if pkl_itself is None:

		#     for file_path in pkl_file_paths:
		#         self.data.extend(pickle.load(open(file_path, 'rb')))
		# else:
		#     self.data = pkl_itself
		self.transform = transform

		self.image_dir = image_dir

		self.num_classes = num_classes

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_data = self.data[idx]
		img_path = img_data['img_path']
		# Trim unnecessary paths

		idx = img_path.split('/').index('CUB_200_2011')
		img_path = '/'.join([self.image_dir] + img_path.split('/')[idx+1:])
		img = Image.open(img_path).convert('RGB')

		class_label = img_data['class_label']
		if self.transform:
			img = self.transform(img)

		return img, class_label


def CUB_iid(dataset,k):
	num_items = int(len(dataset) / k)

	dict_users, all_idxs = {}, [i for i in range(len(dataset))]

	for i in range(k):
		dict_users[i] = set(np.random.choice(all_idxs,num_items,replace=False))#select #num_items indexs from all indexs
		# all_idxs.pop(dict_users[i])
		all_idxs = list(set(all_idxs) - dict_users[i])#remove idx selected

	return dict_users  
def load_cub_data_fl(args):
	filepath = os.path.join(CUB_ATTRIBUTE_DIR,'train.pkl')
	train_data = np.array(pickle.load(open(os.path.join(CUB_ATTRIBUTE_DIR,'train.pkl'),'rb')))
	test_data = np.array(pickle.load(open(os.path.join(CUB_ATTRIBUTE_DIR,'test.pkl'),'rb')))
	val_data = np.array(pickle.load(open(os.path.join(CUB_ATTRIBUTE_DIR,'val.pkl'),'rb')))
	
	train_dict_user = CUB_iid(train_data,args.n_clients)
	test_dict_user = CUB_iid(test_data,args.n_clients)
	val_dict_user = CUB_iid(val_data,args.n_clients)
	normalizer = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
	resol = 224
	transform = transforms.Compose([
				transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
				transforms.RandomResizedCrop(resol),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(), 
				normalizer
				])
	dataset_users = {}
	for i in range(args.n_clients):
		train_index = np.array(list(train_dict_user[i]))
		# print (train_index)
		train_data_client = train_data[train_index]
		train_dataset_client = CUBDataset(train_data_client,CUB_DATA_DIR,200,transform=transform)
				
		test_index = np.array(list(test_dict_user[i]))
		test_data_client = test_data[test_index]
		test_dataset_client = CUBDataset(test_data_client,CUB_DATA_DIR,200,transform=transform)

		val_index = np.array(list(val_dict_user[i]))
		val_data_client = val_data[val_index]
		val_dataset_client = CUBDataset(val_data_client,CUB_DATA_DIR,200,transform=transform)

		train_loader = copy.deepcopy(DataLoader(train_dataset_client,batch_size=args.batch_size,shuffle=True,drop_last=True))
		test_loader = copy.deepcopy(DataLoader(test_dataset_client,batch_size=args.batch_size,shuffle=True,drop_last=True))
		val_loader = copy.deepcopy(DataLoader(val_dataset_client,batch_size=args.batch_size,shuffle=True,drop_last=True))

		dataset_users[i] = [train_loader,test_loader,val_loader]
		
	classes = open(os.path.join(CUB_DATA_DIR, "classes.txt")).readlines()
	classes = [a.split(".")[1].strip() for a in classes]
	idx_to_class = {i: classes[i] for i in range(200)}
	classes = [classes[i] for i in range(200)]

	return dataset_users,idx_to_class,classes

def load_derm_data_fl(args,preprocess):
	import pandas as pd
	from glob import glob
	np.random.seed(args.seed)
	id_to_lesion = {
	'nv': 'Melanocytic nevi',
	'mel': 'dermatofibroma',
	'bkl': 'Benign keratosis-like lesions ',
	'bcc': 'Basal cell carcinoma',
	'akiec': 'Actinic keratoses',
	'vasc': 'Vascular lesions',
	'df': 'Dermatofibroma'}

	benign_malignant = {
	'nv': 'benign',
	'mel': 'malignant',
	'bkl': 'benign',
	'bcc': 'malignant',
	'akiec': 'benign',
	'vasc': 'benign',
	'df': 'benign'}

	df = pd.read_csv(os.path.join(HAM10K_DATA_DIR,'HAM10000_metadata.csv'))
	all_image_paths = glob(os.path.join(HAM10K_DATA_DIR, '*', '*.jpg'))
	id_to_path = {os.path.splitext(os.path.basename(x))[0] : x for x in all_image_paths}

	def path_getter(id):
		if id in id_to_path:
			return id_to_path[id] 
		else:
			return  "-1"
	
	df['path'] = df['image_id'].map(path_getter)
	df = df[df.path != "-1"] 
	df['dx_name'] = df['dx'].map(lambda id: id_to_lesion[id])
	df['benign_or_malignant'] = df["dx"].map(lambda id: benign_malignant[id])
	class_to_idx = {"benign": 0, "malignant": 1}

	df['y'] = df["benign_or_malignant"].map(lambda id: class_to_idx[id])

	idx_to_class = {v: k for k, v in class_to_idx.items()}
	
	#df = df.groupby("y", group_keys=False).apply(pd.DataFrame.sample, 1000)
	_, df_val = train_test_split(df, test_size=0.20, random_state=args.seed, stratify=df["dx"])
	df_train = df[~df.image_id.isin(df_val.image_id)]
	df_test = df_val
	train_dict_user = CUB_iid(df_train,args.n_clients)
	test_dict_user = CUB_iid(df_test,args.n_clients)
	val_dict_user = CUB_iid(df_val,args.n_clients)
	dataset_users = {}
	for i in range(args.n_clients):
		print(f"Train, Val of client {i}: {df_train.shape}, {df_val.shape}")
		train_index = list(train_dict_user[i])
		train_data_client = df_train.iloc[train_index]
		train_dataset_client = DermDataset(train_data_client,preprocess)

		test_index = list(test_dict_user[i])
		test_data_client = df_test.iloc[test_index]
		test_dataset_client = DermDataset(test_data_client,preprocess)

		val_index = list(val_dict_user[i])
		val_data_client = df_val.iloc[val_index]
		val_dataset_client = DermDataset(val_data_client,preprocess)

		train_loader = copy.deepcopy(DataLoader(train_dataset_client,batch_size=args.batch_size,shuffle=True,drop_last=True))
		test_loader = copy.deepcopy(DataLoader(test_dataset_client,batch_size=args.batch_size,shuffle=True,drop_last=True))
		val_loader = copy.deepcopy(DataLoader(val_dataset_client,batch_size=args.batch_size,shuffle=True,drop_last=True))

		dataset_users[i] = [train_loader,test_loader,val_loader]	


	class_to_idx = {v:k for k,v in idx_to_class.items()}
	classes = list(class_to_idx.keys())
	return dataset_users,idx_to_class,classes

def load_cifar10_fl(args,preprocess):
	import ssl
	ssl._create_default_https_context = ssl._create_unverified_context
	# if args.dataset == "cifar10":
	trainset = datasets.CIFAR10(root="data/cifar10/train", train=True,
								download=True, transform=preprocess)
	testset = datasets.CIFAR10(root="data/cifar10/test", train=False,
								download=True, transform=preprocess)
	valset = testset
	train_dict_user = CUB_iid(trainset,args.n_clients)
	test_dict_user = CUB_iid(testset,args.n_clients)
	val_dict_user = CUB_iid(testset,args.n_clients)

	dataset_users = {}
	for i in range(args.n_clients):
		print(f"Train, Val of client {i}: {len(train_dict_user[i])}, {len(test_dict_user[i])}")
		train_sampler = SubsetRandomSampler(list(train_dict_user[i]))
		train_dataset_client = copy.deepcopy(DataLoader(trainset,batch_size=args.batch_size,drop_last=True,sampler=train_sampler))

		test_sampler = SubsetRandomSampler(list(test_dict_user[i]))
		test_dataset_client = copy.deepcopy(DataLoader(testset,batch_size=args.batch_size,drop_last=True,sampler=test_sampler))

		val_sampler = SubsetRandomSampler(list(val_dict_user[i]))
		val_dataset_client = copy.deepcopy(DataLoader(valset,batch_size=args.batch_size,drop_last=True,sampler=val_sampler))

		dataset_users[i] = [train_dataset_client,test_dataset_client,val_dataset_client]


	classes = trainset.classes
	class_to_idx = {c: i for (i,c) in enumerate(classes)}
	idx_to_class = {v: k for k, v in class_to_idx.items()}
	return dataset_users, idx_to_class,classes
    

def load_cifar100_fl(args,preprocess):
	import ssl
	ssl._create_default_https_context = ssl._create_unverified_context
	# if args.dataset == "cifar10":
	trainset = datasets.CIFAR100(root="data/cifar100/train", train=True,
								download=True, transform=preprocess)
	testset = datasets.CIFAR100(root="data/cifar100/test", train=False,
								download=True, transform=preprocess)
	valset = testset
	train_dict_user = CUB_iid(trainset,args.n_clients)
	test_dict_user = CUB_iid(testset,args.n_clients)
	val_dict_user = CUB_iid(testset,args.n_clients)

	dataset_users = {}
	for i in range(args.n_clients):
		print(f"Train, Val of client {i}: {len(train_dict_user[i])}, {len(test_dict_user[i])}")
		train_sampler = SubsetRandomSampler(list(train_dict_user[i]))
		train_dataset_client = copy.deepcopy(DataLoader(trainset,batch_size=args.batch_size,drop_last=True,sampler=train_sampler))

		test_sampler = SubsetRandomSampler(list(test_dict_user[i]))
		test_dataset_client = copy.deepcopy(DataLoader(testset,batch_size=args.batch_size,drop_last=True,sampler=test_sampler))

		val_sampler = SubsetRandomSampler(list(val_dict_user[i]))
		val_dataset_client = copy.deepcopy(DataLoader(valset,batch_size=args.batch_size,drop_last=True,sampler=val_sampler))


		dataset_users[i] = [train_dataset_client,test_dataset_client,val_dataset_client]

	classes = trainset.classes
	class_to_idx = {c: i for (i,c) in enumerate(classes)}
	idx_to_class = {v: k for k, v in class_to_idx.items()}
	return dataset_users,idx_to_class,classes

def load_data(args,preprocess):
	if args.dataset_name == 'cub':
		return load_cub_data_fl(args)
	elif args.dataset_name == 'cifar10':
		return load_cifar10_fl(args,preprocess)
	elif args.dataset_name == 'cifar100':
		return load_cifar100_fl(args,preprocess)
	elif args.dataset_name == 'ham10000':
		return load_derm_data_fl(args,preprocess)
if __name__ == "__main__":
	load_cub_data_fl(5)
	