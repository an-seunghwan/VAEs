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
import torch.nn.functional as F

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
from modules.model_downstream import (
    Classifier
)

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
    tags=["CausalVAE", "DistributionalRobustness"],
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
    artifact = wandb.use_artifact('anseunghwan/CausalDisentangled/DRmodel_{}:v{}'.format('CausalVAE', args["num"]), type='model')
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
        lvae.load_state_dict(torch.load(model_dir + '/DRmodel_{}.pth'.format('CausalVAE')))
    else:
        lvae.load_state_dict(torch.load(model_dir + '/DRmodel_{}.pth'.format('CausalVAE'), 
                                        map_location=torch.device('cpu')))
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
    
    class TestDataset(Dataset): 
        def __init__(self, args):
            foldername = 'pendulum_DR'
            self.name = ['light', 'angle', 'length', 'position', 'background', 'target']
            test_imgs = [x for x in os.listdir('./modules/causal_data/{}/test'.format(foldername)) if x.endswith('png')]

            test_x = []
            for i in tqdm.tqdm(range(len(test_imgs)), desc="test data loading"):
                test_x.append(np.array(
                    Image.open("./modules/causal_data/{}/test/{}".format(foldername, test_imgs[i])).resize((args["image_size"], args["image_size"]))
                    )[:, :, :3])
            self.x_data = np.array(test_x).astype(float) / 255.

            label = np.array([x[:-4].split('_')[1:] for x in test_imgs]).astype(float)
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
    test_dataset = TestDataset(args)
    #%%
    """training dataset"""
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    targets = []
    representations = []
    for x_batch, y_batch in tqdm.tqdm(iter(dataloader)):
        if args["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.no_grad():
            _, _, _, _, _, _, f_z1, _, _, _ = lvae.encode(x_batch, y_batch[:, :5], sample=False)
        targets.append(y_batch)
        representations.append(f_z1.squeeze(dim=-1))
        
    targets = torch.cat(targets, dim=0)
    background = targets[:, [-2]]
    targets = targets[:, [-1]]
    representations = torch.cat(representations, dim=0)[:, :4]
    
    downstream_dataset = TensorDataset(representations, background, targets)
    downstream_dataloader = DataLoader(downstream_dataset, batch_size=64, shuffle=True)
    #%%
    """test dataset"""
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    test_targets = []
    test_representations = []
    for x_batch, y_batch in tqdm.tqdm(iter(test_dataloader)):
        if args["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.no_grad():
            _, _, _, _, _, _, f_z1, _, _, _ = lvae.encode(x_batch, y_batch[:, :5], sample=False)
        test_targets.append(y_batch)
        test_representations.append(f_z1.squeeze(dim=-1))
        
    test_targets = torch.cat(test_targets, dim=0)
    background = test_targets[:, [-2]]
    test_targets = test_targets[:, [-1]]
    test_representations = torch.cat(test_representations, dim=0)[:, :4]
    
    test_downstream_dataset = TensorDataset(test_representations, background, test_targets)
    test_downstream_dataloader = DataLoader(test_downstream_dataset, batch_size=64, shuffle=True)
    #%%
    accuracy_train = []
    worst_accuracy_train = []
    accuracy_test = []
    worst_accuracy_test = []
    for repeat_num in range(10): # repeated experiments
    
        downstream_classifier = Classifier(4, device)
        downstream_classifier = downstream_classifier.to(device)
        
        optimizer = torch.optim.Adam(
            downstream_classifier.parameters(), 
            lr=0.005
        )
        
        downstream_classifier.train()
        
        for epoch in range(200):
            logs = {
                'loss': [], 
            }
            
            for (x_batch, background_batch, y_batch) in iter(downstream_dataloader):
                
                if args["cuda"]:
                    x_batch = x_batch.cuda()
                    background_batch = background_batch.cuda()
                    y_batch = y_batch.cuda()
                
                # with torch.autograd.set_detect_anomaly(True):    
                optimizer.zero_grad()
                
                pred = downstream_classifier(x_batch)
                loss = F.binary_cross_entropy(pred, y_batch, reduction='none').mean()
                
                loss_ = []
                loss_.append(('loss', loss))
                
                loss.backward()
                optimizer.step()
                    
                """accumulate losses"""
                for x, y in loss_:
                    logs[x] = logs.get(x) + [y.item()]
            
            # accuracy
            with torch.no_grad():
                """train accuracy"""
                train_correct = 0
                worst_train_correct = 0
                worst_count = 0
                for (x_batch, background_batch, y_batch) in iter(downstream_dataloader):
                    
                    if args["cuda"]:
                        x_batch = x_batch.cuda()
                        background_batch = background_batch.cuda()
                        y_batch = y_batch.cuda()
                    
                    pred = downstream_classifier(x_batch)
                    pred = (pred > 0.5).float()
                    train_correct += (pred == y_batch).float().sum().item()
                    
                    opposite = torch.where((background_batch - y_batch) != 0)[0]
                    worst_train_correct += (pred[opposite] == y_batch[opposite]).float().sum().item()
                    worst_count += len(opposite)
                    
                train_correct /= downstream_dataset.__len__()
                worst_train_correct /= worst_count
                
                """test accuracy"""
                test_correct = 0
                worst_test_correct = 0
                worst_count = 0
                for (x_batch, background_batch, y_batch) in iter(test_downstream_dataloader):
                    
                    if args["cuda"]:
                        x_batch = x_batch.cuda()
                        background_batch = background_batch.cuda()
                        y_batch = y_batch.cuda()
                    
                    pred = downstream_classifier(x_batch)
                    pred = (pred > 0.5).float()
                    test_correct += (pred == y_batch).float().sum().item()
                    
                    opposite = torch.where((background_batch - y_batch) != 0)[0]
                    worst_test_correct += (pred[opposite] == y_batch[opposite]).float().sum().item()
                    worst_count += len(opposite)
                    
                test_correct /= test_downstream_dataset.__len__()
                worst_test_correct /= worst_count
                
            wandb.log({x : np.mean(y) for x, y in logs.items()})
            wandb.log({'AvgTrainACC(%)' : train_correct * 100})
            wandb.log({'AvgTestACC(%)' : test_correct * 100})
            wandb.log({'WorstTrainACC(%)' : worst_train_correct * 100})
            wandb.log({'WorstTestACC(%)' : worst_test_correct * 100})
        
        print_input = "[Repeat {:02d}]".format(repeat_num + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print_input += ', AvgTrainACC: {:.2f}%'.format(train_correct * 100)
        print_input += ', AvgTestACC: {:.2f}%'.format(test_correct * 100)
        print_input += ', WorstTrainACC: {:.2f}%'.format(worst_train_correct * 100)
        print_input += ', WorstTestACC: {:.2f}%'.format(worst_test_correct * 100)
        print(print_input)
        
        # log accuracy
        accuracy_train.append(train_correct)
        worst_accuracy_train.append(worst_train_correct)
        accuracy_test.append(test_correct)
        worst_accuracy_test.append(worst_test_correct)
    #%%
    """log Accuracy"""
    if not os.path.exists('./assets/robustness/'): 
        os.makedirs('./assets/robustness/')
    with open('./assets/robustness/DR_{}_{}.txt'.format("CausalVAE",  args['num']), 'w') as f:
        f.write('train average accuracy: {:.4f}\n'.format(np.array(accuracy_train).mean()))
        f.write('train worst accuracy: {:.4f}\n'.format(np.array(worst_accuracy_train).mean()))
        f.write('test average accuracy: {:.4f}\n'.format(np.array(accuracy_test).mean()))
        f.write('test worst accuracy: {:.4f}\n'.format(np.array(worst_accuracy_test).mean()))
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%