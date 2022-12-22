#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
# plt.switch_backend('agg')

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.simulation import (
    set_random_seed,
)

from utils.model import (
    VAE,
    Discriminator
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
    project="VAE", 
    entity="anseunghwan",
    tags=["InfoMax", "MNIST", "EDA"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=7, 
                        help='model version')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=True)) # default configuration
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/VAE/InfoMax:v{}'.format(config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    
    """dataset"""
    # training_dataset = datasets.MNIST('./assets/MNIST', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('./assets/MNIST', train=False, download=True, transform=transforms.ToTensor())

    # dataloader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
    
    model = VAE(config["z_dim"], device).to(device)
    discriminator = Discriminator(config["z_dim"])
    if config["cuda"]:
        model.load_state_dict(torch.load(model_dir + '/model.pth'))
        discriminator.load_state_dict(torch.load(model_dir + '/discriminator.pth'))
    else:
        model.load_state_dict(torch.load(model_dir + '/model.pth', map_location=torch.device('cpu')))
        discriminator.load_state_dict(torch.load(model_dir + '/discriminator.pth', map_location=torch.device('cpu')))
    #%%
    """posterior variance"""
    logvars = []
    for (x_batch, y_batch) in tqdm.tqdm(iter(testloader)):
            
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        mu, logvar = model.get_posterior(x_batch)
        logvars.append(logvar)
    logvars = torch.cat(logvars, dim=0)
    variances = torch.exp(logvars).mean(axis=0)
    
    fig = plt.figure(figsize=(5, 3))
    plt.bar(np.arange(config["z_dim"]), variances.detach().numpy(),
            width=0.2)
    plt.xticks(np.arange(config["z_dim"]), np.arange(config["z_dim"]) + 1)
    plt.ylabel('posterior variance', fontsize=12)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('{}/posteriorvariance.png'.format(model_dir), bbox_inches='tight')
    plt.show()
    plt.close()
    #%%
    imgs = []
    for k in [1, 4, 7]:
        imgs.append(Image.open("./artifacts/InfoMax:v{}/posteriorvariance.png".format(k)))
    
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    for i, img in enumerate(imgs):
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.tight_layout()
    plt.savefig('./assets/mutualinfo_effect.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    wandb.log({'mutual information effect': wandb.Image(fig)})
    #%%
    wandb.run.finish()
    #%%