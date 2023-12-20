

from PIL import Image
from torch.utils.data import Dataset, DataLoader
N_ATTRIBUTES = 312
import torch
from .param import *
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
    

def get_concept_dicts(metadata):
    n_concepts = len(metadata[0]["attribute_label"])
    concept_info = {c: {1: [], 0: []} for c in range(n_concepts)}
    for im_data in metadata:
        for c, label in enumerate(im_data["attribute_label"]):
            # print(c)
            img_path = im_data["img_path"]            
            idx = img_path.split('/').index('CUB_200_2011')
            img_path = '/'.join([CUB_DATA_DIR] + img_path.split('/')[idx+1:])
            concept_info[c][label].append(img_path)
    return concept_info