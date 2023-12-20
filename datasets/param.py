
import os
CUB_DATA_DIR = "data/CUB_200_2011"
CUB_ATTRIBUTE_DIR = "datasets/class_attr_data_10"


DERM7_FOLDER = "data/Derm7pt/"
DERM7_META = os.path.join(DERM7_FOLDER, "meta", "meta.csv")
DERM7_TEST_IDX = os.path.join(DERM7_FOLDER,'meta','test_indexes.csv')
DERM7_TRAIN_IDX = os.path.join(DERM7_FOLDER, "meta", "train_indexes.csv")
DERM7_VAL_IDX = os.path.join(DERM7_FOLDER, "meta", "valid_indexes.csv")

HAM10K_DATA_DIR = "data/ham10000"

BRODEN_CONCEPTS = "data/broden"

backbone_name = "resnet18_cifar100"
dataset_name = "cub"
out_dir = "saved_models"
device = 'cuda'
seed = 1
num_workers = 1
concept_batch_size = 10
C = [0.01,0.1]
n_samples_concept = 50

K = 10