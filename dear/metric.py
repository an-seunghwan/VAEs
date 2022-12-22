#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import sys
import random
import tqdm
from PIL import Image

import torch
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import modules
from modules.model import *
from modules.sagan import *
from modules.causal_model import *
from modules.viz import (
    viz_graph,
    viz_heatmap,
)
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

wandb.init(
    project="CausalDisentangled", 
    entity="anseunghwan",
    tags=["Metric"],
)
#%%
import argparse
def get_args(debug):
	parser = argparse.ArgumentParser('parameters')
 
	parser.add_argument('--num', type=int, default=0, 
						help='model version')

	if debug:
		return parser.parse_args(args=[])
	else:    
		return parser.parse_args()
#%%
# import yaml
# def load_config(args):
#     config_path = "./config/{}.yaml".format(args["dataset"])
#     with open(config_path, 'r') as config_file:
#         config = yaml.load(config_file, Loader=yaml.FullLoader)
#     for key in args.keys():
#         if key in config.keys():
#             args[key] = config[key]
#     return args
#%%
def main():
    #%%
    
    args = vars(get_args(debug=False))
    
    artifact = wandb.use_artifact('anseunghwan/CausalDisentangled/model_DEAR:v{}'.format(args["num"]), type='model')
    for key, item in artifact.metadata.items():
        args[key] = item
    args["cuda"] = torch.cuda.is_available()
    wandb.config.update(args)
    
    if 'pendulum' in args["dataset"]:
        label_idx = range(4)
    else:
        if args["labels"] == 'smile':
            label_idx = [31, 20, 19, 21, 23, 13]
        elif args["labels"] == 'age':
            label_idx = [39, 20, 28, 18, 13, 3]
        else:
            raise NotImplementedError("Not supported structure.")
    num_label = len(label_idx)

    random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    if args["cuda"]:
        torch.cuda.manual_seed(args["seed"])
    global device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    #%%
    """dataset"""
    if args["dataset"] == "pendulum":
        class CustomDataset(Dataset): 
            def __init__(self, args):
                foldername = 'pendulum_real'
                train_imgs = [x for x in os.listdir('./modules/causal_data/{}/train'.format(foldername)) if x.endswith('png')]
                train_x = []
                for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
                    train_x.append(np.transpose(
                        np.array(
                        Image.open("./modules/causal_data/{}/train/{}".format(foldername, train_imgs[i])).resize((args["image_size"], args["image_size"]))
                        )[:, :, :3], (2, 0, 1)))
                self.x_data = (np.array(train_x).astype(float) - 127.5) / 127.5
                
                label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
                label = label[:, :4]
                label = label - label.mean(axis=0)
                self.std = label.std(axis=0)
                """bounded label: normalize to (0, 1)"""
                if args["sup_type"] == 'ce': 
                    label = (label - label.min(axis=0)) / (label.max(axis=0) - label.min(axis=0))
                elif args["sup_type"] == 'l2': 
                    label = (label - label.mean(axis=0)) / label.std(axis=0)
                self.y_data = label
                self.name = ['light', 'angle', 'length', 'position']

            def __len__(self): 
                return len(self.x_data)

            def __getitem__(self, idx): 
                x = torch.FloatTensor(self.x_data[idx])
                y = torch.FloatTensor(self.y_data[idx])
                return x, y
        
        dataset = CustomDataset(args)
        train_loader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=False)
    
    elif args["dataset"] == "celeba": 
        train_loader = None
        trans_f = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize((args["image_size"], args["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        data_dir = './utils/causal_data/celeba'
        if not os.path.exists(data_dir): 
            os.makedirs(data_dir)
        train_set = datasets.CelebA(data_dir, split='train', download=True, transform=trans_f)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args["batch_size"], 
                                                    shuffle=False, pin_memory=False,
                                                    drop_last=True, num_workers=0)
    #%%
    if 'scm' in args["prior"]:
        A = torch.zeros((num_label, num_label))
        if args["labels"] == 'smile':
            A[0, 2:6] = 1
            A[1, 4] = 1
        elif args["labels"] == 'age':
            A[0, 2:6] = 1
            A[1, 2:4] = 1
        elif args["labels"] == 'pend':
            A[0, 2:4] = 1
            A[1, 2:4] = 1
    else:
        A = None
    #%%
    """model load"""
    model_dir = artifact.download()
    model = BGM(
        args["latent_dim"], 
        args["g_conv_dim"], 
        args["image_size"],
        args["enc_dist"], 
        args["enc_arch"], 
        args["enc_fc_size"], 
        args["enc_noise_dim"], 
        args["dec_dist"],
        args["prior"], 
        num_label, 
        A
    )
    if args["cuda"]:
        model.load_state_dict(torch.load(model_dir + '/model_DEAR.pth'))
        model = model.to(device)
    else:
        model.load_state_dict(torch.load(model_dir + '/model_DEAR.pth', 
                                        map_location=torch.device('cpu')))
    #%%
    """import baseline classifier"""
    artifact = wandb.use_artifact('anseunghwan/CausalDisentangled/CDMClassifier:v{}'.format(0), type='model')
    model_dir = artifact.download()
    from modules.model_classifier import Classifier
    """masking"""
    mask = []
    # light
    m = torch.zeros(args["image_size"], args["image_size"], 3)
    m[:20, ...] = 1
    mask.append(m)
    # angle
    m = torch.zeros(args["image_size"], args["image_size"], 3)
    m[20:51, ...] = 1
    mask.append(m)
    # shadow
    m = torch.zeros(args["image_size"], args["image_size"], 3)
    m[51:, ...] = 1
    mask.append(m)
    m = torch.zeros(args["image_size"], args["image_size"], 3)
    m[51:, ...] = 1
    mask.append(m)
    
    args["node"] = 4
        
    classifier = Classifier(mask, args, device) 
    if args["cuda"]:
        classifier.load_state_dict(torch.load(model_dir + '/CDMClassifier.pth'))
    else:
        classifier.load_state_dict(torch.load(model_dir + '/CDMClassifier.pth', map_location=torch.device('cpu')))
    #%%
    """latent range"""
    latents = []
    for x_batch, y_batch in tqdm.tqdm(iter(train_loader)):
        if args["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.no_grad(): # out of memory issue
            latent = model.encode(x_batch, mean=True)
        latents.append(latent)
    
    latents = torch.cat(latents, dim=0)
    
    latent_min = latents.cpu().numpy().min(axis=0)
    latent_max = latents.cpu().numpy().max(axis=0)
    #%%
    """metric"""
    dim = 4
    CDM_dict_lower = {x:[] for x in dataset.name}
    CDM_dict_upper = {x:[] for x in dataset.name}
    for s in ['light', 'angle', 'length', 'position']:
        for c in ['light', 'angle', 'length', 'position']:
            CDM_lower = 0
            CDM_upper = 0
            
            dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=False) 
            for x_batch, y_batch in tqdm.tqdm(iter(dataloader), desc="{} | {}".format(c, s)):
                if args["cuda"]:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()

                with torch.no_grad():
                    z = model.encode(x_batch, mean=True)
                    z_inv = model.prior.enc_nlr(z)
                    z_eps = model.prior.get_eps(z_inv)

                    do_index = dataset.name.index(s)
                    min_ = latent_min[do_index]
                    max_ = latent_max[do_index]
                    
                    score = []
                    for val in [min_, max_]:
                        z_new = z.clone()
                        z_new[:, do_index] = torch.tensor(val)
                        z_new_inv = model.prior.enc_nlr(z_new[:, :dim])
                        
                        for j in range(dim):
                            if j == do_index:
                                continue
                            else:
                                z_new_inv[:, j] = torch.matmul(z[:, :j], model.prior.A[:j, j]) + z_eps[:, j]

                        label_z = model.prior.prior_nlr(z_new_inv[:, :dim])
                        other_z = z[:, dim:]
                        
                        xhat = model.decoder(torch.cat([label_z, other_z], dim=1))
                        xhat = xhat.permute(0, 2, 3, 1)
                        
                        """factor classification"""
                        score.append(torch.sigmoid(classifier(xhat))[:, dataset.name.index(c)])
                    
                    CDM_lower += (score[0] - score[1]).sum()
                    CDM_upper += (score[0] - score[1]).abs().sum()
                    
            CDM_lower /= dataset.__len__()
            CDM_upper /= dataset.__len__()
            CDM_dict_lower[s] = CDM_dict_lower.get(s) + [(c, CDM_lower.abs().item())]
            CDM_dict_upper[s] = CDM_dict_upper.get(s) + [(c, CDM_upper.item())]
    #%%
    CDM_mat_lower = np.zeros((dim, dim))
    for i, c in enumerate(dataset.name):
        CDM_mat_lower[i, :] = [x[1] for x in CDM_dict_lower[c]]
    CDM_mat_upper = np.zeros((dim, dim))
    for i, c in enumerate(dataset.name):
        CDM_mat_upper[i, :] = [x[1] for x in CDM_dict_upper[c]]
    
    fig = viz_heatmap(np.flipud(CDM_mat_lower), size=(7, 7))
    wandb.log({'CDM(lower)': wandb.Image(fig)})
    fig = viz_heatmap(np.flipud(CDM_mat_upper), size=(7, 7))
    wandb.log({'CDM(upper)': wandb.Image(fig)})
    
    if not os.path.exists('./assets/CDM/'): 
        os.makedirs('./assets/CDM/')
    # save as csv
    df = pd.DataFrame(CDM_mat_lower.round(3), columns=dataset.name[:4], index=dataset.name[:4])
    df.to_csv('./assets/CDM/lower_{}_{}.csv'.format('DEAR', args['num']))
    df = pd.DataFrame(CDM_mat_upper.round(3), columns=dataset.name[:4], index=dataset.name[:4])
    df.to_csv('./assets/CDM/upper_{}_{}.csv'.format('DEAR', args['num']))
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%