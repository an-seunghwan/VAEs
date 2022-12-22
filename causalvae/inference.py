#%%
#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
#%%
"""
Reference:
[1]: https://github.com/huawei-noah/trustworthyAI/blob/master/research/CausalVAE/inference_pendeulum.py
"""
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from torch.utils import data
import torch.utils.data as Data
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import math
import time
import matplotlib.pyplot as plt
import random
from pprint import pprint
from PIL import Image
import os
import tqdm

from modules.util import _h_A
import modules.util as ut
from modules.mask_vae_pendulum import CausalVAE
from modules.viz import (
    viz_graph,
    viz_heatmap,
)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
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
    tags=["CausalVAE", "Inference"],
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
def main():
    #%%
    
    args = vars(get_args(debug=False)) # default configuration

    """model load"""
    artifact = wandb.use_artifact('anseunghwan/CausalDisentangled/model_{}:v{}'.format('CausalVAE', args["num"]), type='model')
    for key, item in artifact.metadata.items():
        args[key] = item
    pprint(args)
    model_dir = artifact.download()
    
    args["cuda"] = torch.cuda.is_available()
    global device
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    wandb.config.update(args)

    ut.set_random_seed(args["seed"])
    torch.manual_seed(args["seed"])
    if args["cuda"]:
        torch.cuda.manual_seed(args["seed"])

    lvae = CausalVAE(z_dim=args["z_dim"], z1_dim=args["z1_dim"], z2_dim=args["z2_dim"], 
                  	device=device, image_size=args["image_size"]).to(device)
    if args["cuda"]:
        lvae.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format('CausalVAE')))
    else:
        lvae.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format('CausalVAE'), map_location=torch.device('cpu')))
    #%%
    """dataset"""
    class CustomDataset(Dataset): 
        def __init__(self, args):
            foldername = 'pendulum_real'
            train_imgs = [x for x in os.listdir('./modules/causal_data/{}/train'.format(foldername)) if x.endswith('png')]

            train_x = []
            for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
                train_x.append(np.array(
                    Image.open("./modules/causal_data/{}/train/{}".format(foldername, train_imgs[i])).resize((args["image_size"], args["image_size"]))
                    )[:, :, :3])
            self.x_data = np.array(train_x).astype(float) / 255.

            label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
            label = label[:, :4]
            self.std = label.std(axis=0)
            """label standardization"""
            if args["label_standardization"]: 
                label -= label.mean(axis=0)
                label /= label.std(axis=0)
            self.y_data = label
            self.name = ['light', 'angle', 'length', 'position']

        def __len__(self): 
            return len(self.x_data)

        def __getitem__(self, idx): 
            x = torch.FloatTensor(self.x_data[idx])
            y = torch.FloatTensor(self.y_data[idx])
            return x, y
    
    dataset = CustomDataset(args)
    #%%
    """estimated causal matrix"""
    print('DAG:{}'.format(lvae.dag.A))
    B_est = lvae.dag.A.detach().cpu().numpy()
    fig = viz_heatmap(np.flipud(B_est), size=(7, 7))
    wandb.log({'B_est': wandb.Image(fig)})
    #%%
    """intervention range"""
    decode_m_max = []
    decode_m_min = []
    f_z1_max = []
    f_z1_min = []
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=False)
    for u, l in tqdm.tqdm(dataloader):
        if args["cuda"]:
            u = u.cuda()
            l = l.cuda()
        _, _, decode_m, _, _, _, f_z1, _, _, _ = lvae.encode(u, l, sample=False)
        decode_m_max.append(decode_m.squeeze(dim=-1).max(axis=0)[0])
        decode_m_min.append(decode_m.squeeze(dim=-1).min(axis=0)[0])
        f_z1_max.append(f_z1.squeeze(dim=-1).max(axis=0)[0])
        f_z1_min.append(f_z1.squeeze(dim=-1).min(axis=0)[0])
    decode_m_max = torch.vstack(decode_m_max).max(axis=0)[0]
    decode_m_min = torch.vstack(decode_m_min).min(axis=0)[0]
    f_z1_max = torch.vstack(f_z1_max).max(axis=0)[0]
    f_z1_min = torch.vstack(f_z1_min).min(axis=0)[0]
    
    causal_range = [(decode_m_min[0].item(), decode_m_max[0].item()), 
                    (decode_m_min[1].item(), decode_m_max[1].item()),
                    (f_z1_min[2].item(), f_z1_max[2].item()), 
                    (f_z1_min[3].item(), f_z1_max[3].item())]
    #%%
    """do-intervention"""
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    iter_test = iter(dataloader)
    count = 1
    for _ in range(count):
        u, l = next(iter_test)
    if args["cuda"]:
        u = u.cuda()
        l = l.cuda()
    
    fig, ax = plt.subplots(4, 7, figsize=(7, 4))
    
    for i in range(4): # masking node index
        if i < 2:
            for k, j in enumerate(np.linspace(causal_range[i][0], causal_range[i][1], 7)): # do-intervention value
                _, _, _, _, reconstructed_image, _= lvae.negative_elbo_bound(u, l, i, sample = False, adj=j)
                ax[i, k].imshow(torch.sigmoid(reconstructed_image[0]).detach().cpu().numpy())
                ax[i, k].axis('off')    
        else:
            for k, j in enumerate(np.linspace(causal_range[i][0], causal_range[i][1], 7)): # do-intervention value
                _, _, _, _, reconstructed_image, _= lvae.negative_elbo_bound(u, l, i, sample = False, adj=j)
                ax[i, k].imshow(torch.sigmoid(reconstructed_image[0]).detach().cpu().numpy())
                ax[i, k].axis('off')    
    
    plt.tight_layout()
    plt.savefig('{}/do.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'do intervention ({})'.format(', '.join(dataset.name)): wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%