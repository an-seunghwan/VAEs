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
[1]: https://github.com/huawei-noah/trustworthyAI/blob/master/research/CausalVAE/run_flow.py
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
plt.switch_backend('agg')
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
    tags=["CausalVAE", "DR"],
)
#%%
import argparse
def get_args(debug):
	parser = argparse.ArgumentParser('parameters')
	# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
	parser.add_argument('--seed', type=int, default=2, 
						help='seed for repeatable results')

	parser.add_argument("--z_dim", default=5, type=int,
                        help="the number of latent dimension")
	parser.add_argument("--z1_dim", default=5, type=int,
                        help="the number of latent 1 dimension")
	parser.add_argument("--z2_dim", default=1, type=int,
                        help="the number of latent 2 dimension")
	parser.add_argument('--image_size', default=64, type=int,
						help='width and heigh of image')

	parser.add_argument("--label_standardization", default=True, type=bool,
                        help="If True, normalize additional information label data")
 
	parser.add_argument('--epochs', default=200, type=int,
						help='maximum iteration')
	parser.add_argument('--batch_size', default=128, type=int,
						help='batch size')
	parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')

	if debug:
		return parser.parse_args(args=[])
	else:    
		return parser.parse_args()
#%%
def train(dataloader, lvae, args, optimizer, device):
	logs = {
		'loss': [], 
		'recon': [],
		'KL': [],
		'mask': [],
		# 'DAG': [],
	}

	for u, l in tqdm.tqdm(dataloader):
		if args["cuda"]:
			u = u.cuda()
			l = l.cuda()

		optimizer.zero_grad()

		loss_ = []

		L, kl, rec, mask, reconstructed_image, _ = lvae.negative_elbo_bound(u, l, sample = False)
		# dag_param = lvae.dag.A

		# h_a = _h_A(dag_param, dag_param.size()[0])
		# dag = 3*h_a + 0.5*h_a*h_a
		# L = L + dag
  
		loss_.append(('loss', L))
		loss_.append(('recon', rec))
		loss_.append(('KL', kl))
		loss_.append(('mask', mask))
		# loss_.append(('DAG', dag))

		L.backward()
		"""FIXME: Given true-graph"""
		lvae.dag.set_zero_grad()
		optimizer.step()

		"""accumulate losses"""
		for x, y in loss_:
			logs[x] = logs.get(x) + [y.item()]
    
	return logs, reconstructed_image
#%%
def main():
    #%%
	args = vars(get_args(debug=False)) # default configuration
	pprint(args)

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
	optimizer = torch.optim.Adam(lvae.parameters(), 
								lr=args["lr"], 
								betas=(0.9, 0.999))
	#%%
	"""dataset"""
	class CustomDataset(Dataset): 
		def __init__(self, args):
			foldername = 'pendulum_DR'
			self.name = ['light', 'angle', 'length', 'position', 'background', 'target']
			train_imgs = [x for x in os.listdir('./modules/causal_data/{}/train'.format(foldername)) if x.endswith('png')]
   
			train_x = []
			for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
				train_x.append(np.array(
					Image.open("./modules/causal_data/{}/train/{}".format(foldername, train_imgs[i])).resize((args["image_size"], args["image_size"]))
					)[:, :, :3])
			self.x_data = np.array(train_x).astype(float) / 255.
			
			label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
			label = label[:, :5]
			self.std = label.std(axis=0)
			"""label standardization"""
			if args["label_standardization"]: 
				label[:, :4] -= label[:, :4].mean(axis=0)
				label[:, :4] /= label[:, :4].std(axis=0)
			self.y_data = label

		def __len__(self): 
			return len(self.x_data)

		def __getitem__(self, idx): 
			x = torch.FloatTensor(self.x_data[idx])
			y = torch.FloatTensor(self.y_data[idx])
			return x, y

	dataset = CustomDataset(args)
	dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)
	#%%
	lvae.train()

	for epoch in range(args["epochs"]):
		logs, xhat = train(dataloader, lvae, args, optimizer, device)

		print_input = "[epoch {:03d}]".format(epoch + 1)
		print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
		print(print_input)

		"""update log"""
		wandb.log({x : np.mean(y) for x, y in logs.items()})

		"""estimated causal adjacency matrix"""
		B_est = lvae.dag.A.detach().cpu().numpy()
		fig = viz_heatmap(np.flipud(B_est), size=(7, 7))
		wandb.log({'B_est': wandb.Image(fig)})

		if epoch % 10 == 0:
			plt.figure(figsize=(4, 4))
			for i in range(9):
				plt.subplot(3, 3, i+1)
				plt.imshow(torch.sigmoid(xhat)[i].cpu().detach().numpy())
				plt.axis('off')
			plt.savefig('./assets/tmp_image_{}.png'.format(epoch))
			plt.close()

	"""reconstruction result"""
	fig = plt.figure(figsize=(4, 4))
	for i in range(9):
		plt.subplot(3, 3, i+1)
		plt.imshow(torch.sigmoid(xhat)[i].cpu().detach().numpy())
		plt.axis('off')
	plt.savefig('./assets/recon.png')
	plt.close()
	wandb.log({'reconstruction': wandb.Image(fig)})

	"""model save"""
	torch.save(lvae.state_dict(), './assets/DRmodel_{}.pth'.format('CausalVAE'))
	artifact = wandb.Artifact('DRmodel_{}'.format('CausalVAE'), 
								type='model',
								metadata=args) # description=""
	artifact.add_file('./assets/DRmodel_{}.pth'.format('CausalVAE'))
	wandb.log_artifact(artifact)
    #%%
	wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%