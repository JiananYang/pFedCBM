
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

from datasets.param import *
# from .param import backbone_name,dataset_name,out_dir,device,seed,num_workers,concept_batch_size,n_samples_concept,C
from .param import *
from torch.utils.data import Dataset, DataLoader
from datasets.cub import CUBDataset
N_ATTRIBUTES = 312

def CUB_iid(dataset,k):
	num_items = int(len(dataset) / k)

	dict_users, all_idxs = {}, [i for i in range(len(dataset))]

	for i in range(k):
		dict_users[i] = set(np.random.choice(all_idxs,num_items,replace=False))#select #num_items indexs from all indexs
		# all_idxs.pop(dict_users[i])
		all_idxs = list(set(all_idxs) - dict_users[i])#remove idx selected

	return dict_users  
def load_cub_data_fl(args,n_clients):
	filepath = os.path.join(CUB_ATTRIBUTE_DIR,'train.pkl')
	train_data = np.array(pickle.load(open(os.path.join(CUB_ATTRIBUTE_DIR,'train.pkl'),'rb')))
	test_data = np.array(pickle.load(open(os.path.join(CUB_ATTRIBUTE_DIR,'test.pkl'),'rb')))
	val_data = np.array(pickle.load(open(os.path.join(CUB_ATTRIBUTE_DIR,'val.pkl'),'rb')))
	
	train_dict_user = CUB_iid(train_data,n_clients)
	test_dict_user = CUB_iid(test_data,n_clients)
	val_dict_user = CUB_iid(val_data,n_clients)
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
	for i in range(n_clients):
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
		

	return dataset_users


def load_derm_data_fl(args,n_clients):
	filepath = os.path.join(CUB_ATTRIBUTE_DIR,'train.pkl')
	train_data = np.array(pickle.load(open(os.path.join(CUB_ATTRIBUTE_DIR,'train.pkl'),'rb')))
	test_data = np.array(pickle.load(open(os.path.join(CUB_ATTRIBUTE_DIR,'test.pkl'),'rb')))
	val_data = np.array(pickle.load(open(os.path.join(CUB_ATTRIBUTE_DIR,'val.pkl'),'rb')))
	
	train_dict_user = CUB_iid(train_data,n_clients)
	test_dict_user = CUB_iid(test_data,n_clients)
	val_dict_user = CUB_iid(val_data,n_clients)
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
	for i in range(n_clients):
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
		

	return dataset_users
