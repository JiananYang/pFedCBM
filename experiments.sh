

#concept bank
python FLSVM.py --dataset_name "cub" --backbone_name "resnet18_cub"

python FLSVM.py --dataset_name "broden" --backbone_name "resnet50_cifar"

python FLSVM.py --dataset_name "derm7pt" --backbone_name "ham10000_inception"


#predictor head
python main.py --dataset_name "cub" --concept_dataset_name "cub" --backbone_name "resnet18_cub" --num_classes 200 --n_clients 5

python main.py --dataset_name "cifar10" --concept_dataset_name "broden" --backbone_name "resnet50_cifar" --num_classes 10 --n_clients 5

python main.py --dataset_name "cifar100" --concept_dataset_name "broden" --backbone_name "resnet50_cifar" --num_classes 100 --n_clients 5

python main.py --dataset_name "ham10000" --concept_dataset_name "derm7pt" --backbone_name "ham10000_inception" --num_classes 2 --n_clients 5