#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from utils.simulation import (
    set_random_seed,
)

from utils.model import (
    VQVAE
)
#%%
"""
wandb artifact cache cleanup "1GB"
"""
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
    tags=["VQVAE", "CIFAR-10"]
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    
    parser.add_argument("--num_embeddings", default=256, type=int,
                        help="the size of the discrete latent space(K)")
    parser.add_argument("--embedding_dim", default=64, type=int,
                        help="the dimensionality of the discrete latent space")
    
    parser.add_argument('--epochs', default=10, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    
    parser.add_argument('--beta', default=0.25, type=float,
                        help='coefficient of commitment loss')
    
    parser.add_argument('--fig_show', default=False, type=bool)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def train(trainloader, model, config, optimizer, device):
    logs = {
        'loss': [], 
        'recon': [],
        'VQ': [],
    }
    
    for batch in tqdm.tqdm(iter(trainloader), desc="inner loop"):
        
        batch = batch[0]
        if config["cuda"]:
            batch = batch.cuda()
        # break
        
        with torch.autograd.set_detect_anomaly(True):    
            optimizer.zero_grad()
            
            xhat, _, vq_loss = model(batch)
            
            loss_ = []
            
            """reconstruction"""
            recon = F.mse_loss(xhat, batch)
            loss_.append(('recon', recon))

            """VQ loss"""
            loss_.append(('VQ', vq_loss))
            
            loss = recon + vq_loss
            loss_.append(('loss', loss))
            
            loss.backward()
            optimizer.step()
            
            """accumulate losses"""
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()]
                
    return logs, xhat
#%%
def main():
    config = vars(get_args(debug=False)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
        
    """CIFAR-10 dataset"""
    # URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: 
    # certificate has expired (_ssl.c:1131)>
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"],
                                            shuffle=True, num_workers=0, pin_memory=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    model = VQVAE(config, device, hidden_dims=[128, 256])
    model.to(device)
    # next(model.parameters()).is_cuda

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    
    wandb.watch(model, log_freq=100) # tracking gradients
    model.train()
    
    for epoch in range(config["epochs"]):
        logs, xhat = train(trainloader, model, config, optimizer, device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        
        if epoch % 10 == 0:
            plt.figure(figsize=(4, 4))
            for i in range(9):
                plt.subplot(3, 3, i+1)
                plt.imshow((xhat[i].cpu().permute((1, 2, 0)).detach().numpy() + 1) / 2)
                # plt.imshow((xhat[i].cpu().detach().numpy() + 1) / 2)
                plt.axis('off')
            plt.savefig('./assets/tmp_image_{}.png'.format(epoch))
            plt.close()
    
    """reconstruction result"""
    fig = plt.figure(figsize=(4, 4))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow((xhat[i].cpu().permute((1, 2, 0)).detach().numpy() + 1) / 2)
        # plt.imshow((xhat[i].cpu().detach().numpy() + 1) / 2)
        plt.axis('off')
    plt.savefig('./assets/image.png')
    plt.close()
    wandb.log({'reconstruction': wandb.Image(fig)})
    
    """model save"""
    torch.save(model.state_dict(), './assets/model.pth')
    artifact = wandb.Artifact('model', type='model') # description=""
    artifact.add_file('./assets/model.pth')
    wandb.log_artifact(artifact)
    
    # """model load"""
    # artifact = wandb.use_artifact('anseunghwan/VAE/model:v0', type='model')
    # model_dir = artifact.download()
    # model_ = VQVAE(config, device, hidden_dims=[128, 256])
    # model_.load_state_dict(torch.load(model_dir + '/model.pth'))
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%