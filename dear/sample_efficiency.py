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

from modules.model_downstream import (
    Classifier
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
    tags=["SampleEfficiency"],
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
                self.y_data = label
                self.name = ['light', 'angle', 'length', 'position']

            def __len__(self): 
                return len(self.x_data)

            def __getitem__(self, idx): 
                x = torch.FloatTensor(self.x_data[idx])
                y = torch.FloatTensor(self.y_data[idx])
                return x, y
        
        class TestDataset(Dataset): 
            def __init__(self, args):
                foldername = 'pendulum_real'
                test_imgs = [x for x in os.listdir('./modules/causal_data/{}/test'.format(foldername)) if x.endswith('png')]
                test_x = []
                for i in tqdm.tqdm(range(len(test_imgs)), desc="test data loading"):
                    test_x.append(np.transpose(
                        np.array(
                        Image.open("./modules/causal_data/{}/test/{}".format(foldername, test_imgs[i])).resize((args["image_size"], args["image_size"]))
                        )[:, :, :3], (2, 0, 1)))
                self.x_data = (np.array(test_x).astype(float) - 127.5) / 127.5
                
                label = np.array([x[:-4].split('_')[1:] for x in test_imgs]).astype(float)
                self.y_data = label
                self.name = ['light', 'angle', 'length', 'position']

            def __len__(self): 
                return len(self.x_data)

            def __getitem__(self, idx): 
                x = torch.FloatTensor(self.x_data[idx])
                y = torch.FloatTensor(self.y_data[idx])
                return x, y
        
        dataset = CustomDataset(args)
        test_dataset = TestDataset(args)
    
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
                                                    shuffle=True, pin_memory=False,
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
    beta = torch.tensor([[1, -1, 0.5, -0.5]]).to(device)
    
    """with 100 size of training dataset"""
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    targets_100 = []
    representations_100 = []
    for count, (x_batch, y_batch) in tqdm.tqdm(enumerate(iter(dataloader))):
        if args["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.no_grad():
            mean = model.encode(x_batch, mean=True)
        targets_100.append(y_batch)
        representations_100.append(mean)
        
        count += 1
        if count == 50: break
    targets_100 = torch.cat(targets_100, dim=0)
    logit = torch.matmul(targets_100[:, :-1], beta.t())
    targets_100 = torch.bernoulli(1 / (1 + torch.exp(-logit - 2*torch.sin(logit))))
    representations_100 = torch.cat(representations_100, dim=0)[:, :len(label_idx)]
    
    downstream_dataset_100 = TensorDataset(representations_100, targets_100)
    downstream_dataloader_100 = DataLoader(downstream_dataset_100, batch_size=32, shuffle=True)
    #%%
    """with all training dataset"""
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    targets = []
    representations = []
    for x_batch, y_batch in tqdm.tqdm(iter(dataloader)):
        if args["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.no_grad():
            mean = model.encode(x_batch, mean=True)
        targets.append(y_batch)
        representations.append(mean)
        
    targets = torch.cat(targets, dim=0)
    logit = torch.matmul(targets[:, :-1], beta.t())
    targets = torch.bernoulli(1 / (1 + torch.exp(-logit - 2*torch.sin(logit))))
    representations = torch.cat(representations, dim=0)[:, :len(label_idx)]
    
    downstream_dataset = TensorDataset(representations, targets)
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
            mean = model.encode(x_batch, mean=True)
        test_targets.append(y_batch)
        test_representations.append(mean)
        
    test_targets = torch.cat(test_targets, dim=0)
    logit = torch.matmul(test_targets[:, :-1], beta.t())
    test_targets = torch.bernoulli(1 / (1 + torch.exp(-logit - 2*torch.sin(logit))))
    test_representations = torch.cat(test_representations, dim=0)[:, :len(label_idx)]
    
    test_downstream_dataset = TensorDataset(test_representations, test_targets)
    test_downstream_dataloader = DataLoader(test_downstream_dataset, batch_size=64, shuffle=True)
    #%%
    accuracy = []
    accuracy_100 = []
    for repeat_num in range(10): # repeated experiments
    
        print("Sample Efficiency with 100 labels")
        downstream_classifier_100 = Classifier(len(label_idx), device)
        downstream_classifier_100 = downstream_classifier_100.to(device)
        
        optimizer = torch.optim.Adam(
            downstream_classifier_100.parameters(), 
            lr=0.005
        )
        
        downstream_classifier_100.train()
        
        for epoch in range(100):
            logs = {
                'loss': [], 
            }
            
            for (x_batch, y_batch) in iter(downstream_dataloader_100):
                
                if args["cuda"]:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                
                # with torch.autograd.set_detect_anomaly(True):    
                optimizer.zero_grad()
                
                pred = downstream_classifier_100(x_batch)
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
                for (x_batch, y_batch) in iter(downstream_dataloader_100):
                    
                    if args["cuda"]:
                        x_batch = x_batch.cuda()
                        y_batch = y_batch.cuda()
                    
                    pred = downstream_classifier_100(x_batch)
                    pred = (pred > 0.5).float()
                    train_correct += (pred == y_batch).float().sum().item()
                train_correct /= downstream_dataset_100.__len__()
                
                """test accuracy"""
                test_correct = 0
                for (x_batch, y_batch) in iter(test_downstream_dataloader):
                    
                    if args["cuda"]:
                        x_batch = x_batch.cuda()
                        y_batch = y_batch.cuda()
                    
                    pred = downstream_classifier_100(x_batch)
                    pred = (pred > 0.5).float()
                    test_correct += (pred == y_batch).float().sum().item()
                test_correct /= test_downstream_dataset.__len__()
            
            wandb.log({x : np.mean(y) for x, y in logs.items()})
            wandb.log({'TrainACC(%)_100samples' : train_correct * 100})
            wandb.log({'TestACC(%)_100samples' : test_correct * 100})
        
        print_input = "[Repeat {:02d}]".format(repeat_num + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print_input += ', TrainACC: {:.2f}%'.format(train_correct * 100)
        print_input += ', TestACC: {:.2f}%'.format(test_correct * 100)
        print(print_input)
        
        # log accuracy
        accuracy_100.append(test_correct)
        
        print("Sample Efficiency with all labels")
        downstream_classifier = Classifier(len(label_idx), device)
        downstream_classifier = downstream_classifier.to(device)
        
        optimizer = torch.optim.Adam(
            downstream_classifier.parameters(), 
            lr=0.005
        )
        
        downstream_classifier.train()
        
        for epoch in range(100):
            logs = {
                'loss': [], 
            }
            
            for (x_batch, y_batch) in iter(downstream_dataloader):
                
                if args["cuda"]:
                    x_batch = x_batch.cuda()
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
                for (x_batch, y_batch) in iter(downstream_dataloader):
                    
                    if args["cuda"]:
                        x_batch = x_batch.cuda()
                        y_batch = y_batch.cuda()
                    
                    pred = downstream_classifier(x_batch)
                    pred = (pred > 0.5).float()
                    train_correct += (pred == y_batch).float().sum().item()
                train_correct /= downstream_dataset.__len__()
                
                """test accuracy"""
                test_correct = 0
                for (x_batch, y_batch) in iter(test_downstream_dataloader):
                    
                    if args["cuda"]:
                        x_batch = x_batch.cuda()
                        y_batch = y_batch.cuda()
                    
                    pred = downstream_classifier(x_batch)
                    pred = (pred > 0.5).float()
                    test_correct += (pred == y_batch).float().sum().item()
                test_correct /= test_downstream_dataset.__len__()
            
            wandb.log({x : np.mean(y) for x, y in logs.items()})
            wandb.log({'TrainACC(%)' : train_correct * 100})
            wandb.log({'TestACC(%)' : test_correct * 100})
        
        print_input = "[Repeat {:02d}]".format(repeat_num + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print_input += ', TrainACC: {:.2f}%'.format(train_correct * 100)
        print_input += ', TestACC: {:.2f}%'.format(test_correct * 100)
        print(print_input)
            
        # log accuracy
        accuracy.append(test_correct)
    #%%
    """log Accuracy"""
    sample_efficiency = np.array(accuracy_100).mean() / np.array(accuracy).mean()
    if not os.path.exists('./assets/sample_efficiency/'): 
        os.makedirs('./assets/sample_efficiency/')
    with open('./assets/sample_efficiency/DEAR_{}.txt'.format(args['num']), 'w') as f:
        f.write('100 samples accuracy: {:.4f}\n'.format(np.array(accuracy_100).mean()))
        f.write('all samples accuracy: {:.4f}\n'.format(np.array(accuracy).mean()))
        f.write('sample efficiency: {:.4f}\n'.format(sample_efficiency))
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%