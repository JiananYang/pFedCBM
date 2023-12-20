import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from helpers import makedir

def unpack_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1]
    elif len(batch) == 2:
        return batch
    else:
        raise ValueError()


@torch.no_grad()
def get_projections(args, backbone, posthoc_layer, loader):
    all_projs, all_embs, all_lbls = None, None, None
    for batch in tqdm(loader):
        batch_X, batch_Y = unpack_batch(batch)
        batch_X = batch_X.to(args.device)
        if "clip" in args.backbone_name:
            embeddings = backbone.encode_image(batch_X).detach().float()
        else:
            embeddings = backbone(batch_X)
        # print (posthoc_layer.compute_dist(embeddings))
        projs = posthoc_layer.compute_dist(embeddings)

        if all_embs is None:
            all_embs = embeddings
            all_projs = projs
            all_lbls = batch_Y
        else:
            all_embs = torch.cat([all_embs, embeddings], dim=0)
            all_projs = torch.cat([all_projs, projs], dim=0)
            all_lbls = torch.cat([all_lbls, batch_Y], dim=0)
    return all_embs, all_projs, all_lbls


class EmbDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y
    def __len__(self):
        return len(self.data)


def load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader):
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
    
    # To make it easier to analyize results/rerun with different params, we'll extract the embeddings and save them
    train_file = f"train-embs_{args.dataset}__{args.backbone_name}.pt"
    test_file = f"test-embs_{args.dataset}__{args.backbone_name}.pt"
    train_proj_file = f"train-proj_{args.dataset}__{args.backbone_name}.pt"
    test_proj_file = f"test-proj_{args.dataset}__{args.backbone_name}.pt"
    train_lbls_file = f"train-lbls_{args.dataset}__{args.backbone_name}_lbls.pt"
    test_lbls_file = f"test-lbls_{args.dataset}__{args.backbone_name}_lbls.pt"
    
    outdir = os.path.join(args.out_dir,'FLcub')
    makedir(outdir)
    train_file = os.path.join(outdir, train_file)
    test_file = os.path.join(outdir, test_file)
    train_proj_file = os.path.join(outdir, train_proj_file)
    test_proj_file = os.path.join(outdir, test_proj_file)
    train_lbls_file = os.path.join(outdir, train_lbls_file)
    test_lbls_file = os.path.join(outdir, test_lbls_file)

    # if os.path.exists(train_proj_file):
    #     train_embs = np.load(train_file)
    #     test_embs = np.load(test_file)
    #     train_projs = np.load(train_proj_file)
    #     test_projs = np.load(test_proj_file)
    #     train_lbls = np.load(train_lbls_file)
    #     test_lbls = np.load(test_lbls_file)

    # else:
    train_embs, train_projs, train_lbls = get_projections(args, backbone, posthoc_layer, train_loader)
    test_embs, test_projs, test_lbls = get_projections(args, backbone, posthoc_layer, test_loader)

    torch.save(train_embs,train_file)
    torch.save(test_embs,test_file)
    torch.save(train_projs,train_proj_file)
    torch.save(test_projs,test_proj_file)
    torch.save(train_lbls,train_lbls_file)
    torch.save(test_lbls,test_lbls_file)
    
    return train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls
