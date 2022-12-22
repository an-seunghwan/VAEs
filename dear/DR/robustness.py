#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import sys
import random
import tqdm
from PIL import Image
from scipy.stats.contingency import crosstab

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
    tags=["DEAR", "DistributionalRobustness"],
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
    
    artifact = wandb.use_artifact('anseunghwan/CausalDisentangled/DRmodel_DEAR:v{}'.format(args["num"]), type='model')
    for key, item in artifact.metadata.items():
        args[key] = item
    args["cuda"] = torch.cuda.is_available()
    wandb.config.update(args)
    
    if 'pendulum' in args["dataset"]:
        label_idx = range(5)
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
                foldername = 'pendulum_DR'
                self.name = ['light', 'angle', 'length', 'position', 'background', 'target']
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
                    test_x.append(np.transpose(
                        np.array(
                        Image.open("./modules/causal_data/{}/test/{}".format(foldername, test_imgs[i])).resize((args["image_size"], args["image_size"]))
                        )[:, :, :3], (2, 0, 1)))
                self.x_data = (np.array(test_x).astype(float) - 127.5) / 127.5
                
                label = np.array([x[:-4].split('_')[1:] for x in test_imgs]).astype(float)
                self.y_data = label

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
        model.load_state_dict(torch.load(model_dir + '/DRmodel_DEAR.pth'))
        model = model.to(device)
    else:
        model.load_state_dict(torch.load(model_dir + '/DRmodel_DEAR.pth', 
                                        map_location=torch.device('cpu')))
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
            mean = model.encode(x_batch, mean=True)
        targets.append(y_batch)
        representations.append(mean)
        
    targets = torch.cat(targets, dim=0)
    background = targets[:, [-2]]
    targets = targets[:, [-1]]
    representations = torch.cat(representations, dim=0)[:, :len(label_idx)-1]
    
    downstream_dataset = TensorDataset(representations, background, targets)
    downstream_dataloader = DataLoader(downstream_dataset, batch_size=64, shuffle=True)
    
    print('Train dataset label crosstab:')
    print(crosstab(background.cpu().numpy(), targets.cpu().numpy())[1] / len(targets))
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
    background = test_targets[:, [-2]]
    test_targets = test_targets[:, [-1]]
    test_representations = torch.cat(test_representations, dim=0)[:, :len(label_idx)-1]
    
    test_downstream_dataset = TensorDataset(test_representations, background, test_targets)
    test_downstream_dataloader = DataLoader(test_downstream_dataset, batch_size=64, shuffle=True)
    
    print('Test dataset label crosstab:')
    print(crosstab(background.cpu().numpy(), test_targets.cpu().numpy())[1] / len(test_targets))
    #%%
    accuracy_train = []
    worst_accuracy_train = []
    accuracy_test = []
    worst_accuracy_test = []
    for repeat_num in range(10): # repeated experiments
    
        downstream_classifier = Classifier(len(label_idx)-1, device)
        downstream_classifier = downstream_classifier.to(device)
        
        optimizer = torch.optim.Adam(
            downstream_classifier.parameters(), 
            lr=0.005
        )
        
        downstream_classifier.train()
        
        for epoch in range(500):
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
    with open('./assets/robustness/DR_{}_{}.txt'.format("DEAR", args['num']), 'w') as f:
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