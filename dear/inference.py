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
    tags=["DEAR", "Inference"],
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
    discriminator = BigJointDiscriminator(
        args["latent_dim"], 
        args["d_conv_dim"], 
        args["image_size"],
        args["dis_fc_size"]
    )
    if args["cuda"]:
        model.load_state_dict(torch.load(model_dir + '/model_DEAR.pth'))
        model = model.to(device)
    else:
        model.load_state_dict(torch.load(model_dir + '/model_DEAR.pth', 
                                        map_location=torch.device('cpu')))
    #%%
    """estimated causal matrix"""
    print('DAG:{}'.format(model.prior.A))
    B_est = model.prior.A.detach().cpu().numpy()
    fig = viz_heatmap(np.flipud(B_est), size=(7, 7))
    wandb.log({'B_est': wandb.Image(fig)})
    #%%
    """do-intervention"""
    n = 7
    gap = 3
    traversals = torch.linspace(-gap, gap, steps=n)
    
    dim = model.num_label if model.num_label is not None else model.latent_dim
    z = torch.zeros(1, args["latent_dim"], device=device)
    z = z.expand(n, model.latent_dim)
    
    fig, ax = plt.subplots(4, n, figsize=(n, 4))
    
    for idx in range(dim):
        z_inv = model.prior.enc_nlr(z)
        z_eps = model.prior.get_eps(z_inv)
        
        z_new = z.clone()
        z_new[:, idx] = traversals
        z_new_inv = model.prior.enc_nlr(z_new[:, :dim])
        
        for j in range(dim):
            if j == idx:
                continue
            else:
                z_new_inv[:, j] = torch.matmul(z[:, :j], model.prior.A[:j, j]) + z_eps[:, j]
        
        label_z = model.prior.prior_nlr(z_new_inv[:, :dim])
        other_z = z[:, dim:]
        
        with torch.no_grad():
            xhat = model.decoder(torch.cat([label_z, other_z], dim=1))
            for k in range(n):
                ax[idx, k].imshow((xhat[k].cpu().permute(1, 2, 0) + 1) / 2)
                ax[idx, k].axis('off')
    
    plt.tight_layout()
    plt.savefig('{}/do.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    name = ['light', 'angle', 'length', 'position']
    wandb.log({'do intervention ({})'.format(', '.join(name)): wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%