#%%
"""
Reference:
[1]: https://github.com/xwshen51/DEAR/blob/main/train.py
"""
#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
    tags=["DEAR"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser(description='Disentangled Generative Causal Representation (DEAR)')

    # Data settings
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='pendulum', choices=['celeba', 'pendulum'])

    # Training settings
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr_g', type=float, default=5e-5)
    parser.add_argument('--lr_e', type=float, default=5e-5)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--lr_p', type=float, default=5e-5, help='lr of SCM prior network')
    parser.add_argument('--lr_a', type=float, default=1e-3, help='lr of adjacency matrix')
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--d_steps_per_iter', type=int, default=1, help='how many D updates per iteration')
    parser.add_argument('--g_steps_per_iter', type=int, default=1, help='how many G updates per iteration')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)

    # Model settings
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--sup_coef', type=float, default=1, help='coefficient of the supervised regularizer')
    parser.add_argument('--sup_prop', type=float, default=1, help='proportion of supervised labels')
    parser.add_argument('--sup_type', type=str, default='ce', choices=['ce', 'l2'])
    parser.add_argument('--labels', type=str, default=None, help='name of the underlying structure')

    # Prior settings
    parser.add_argument('--prior', type=str, default='linscm', choices=['gaussian', 'uniform', 'linscm', 'nlrscm'],
                        help='latent prior p_z')

    # Encoder settings
    parser.add_argument('--enc_arch', type=str, default='resnet', choices=['resnet', 'resnet18', 'dcgan'],
                        help='encoder architecture')
    parser.add_argument('--enc_dist', type=str, default='gaussian', choices=['deterministic', 'gaussian', 'implicit'],
                        help='encoder distribution')
    parser.add_argument('--enc_fc_size', type=int, default=1024, help='number of nodes in fc layer of resnet')
    parser.add_argument('--enc_noise_dim', type=int, default=128)
    # Generator settings
    parser.add_argument('--dec_arch', type=str, default='sagan', choices=['sagan', 'dcgan'],
                        help='decoder architecture')
    parser.add_argument('--dec_dist', type=str, default='implicit', choices=['deterministic', 'gaussian', 'implicit'],
                        help='generator distribution')
    parser.add_argument('--g_conv_dim', type=int, default=32, help='base number of channels in encoder and generator')
    # Discriminator settings
    parser.add_argument('--dis_fc_size', type=int, default=512, help='number of nodes in fc layer of joint discriminator')
    parser.add_argument('--d_conv_dim', type=int, default=32, help='base number of channels in discriminator')

    # Output and save
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--sample_every', type=int, default=1)
    parser.add_argument('--sample_every_epoch', type=int, default=1)
    parser.add_argument('--save_model_every', type=int, default=5)
    parser.add_argument('--save_name', type=str, default='')
    parser.add_argument('--save_n_samples', type=int, default=64)
    parser.add_argument('--save_n_recons', type=int, default=32)
    parser.add_argument('--nrow', type=int, default=8)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
import yaml
def load_config(args):
    config_path = "./config/{}.yaml".format(args["dataset"])
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in args.keys():
        if key in config.keys():
            args[key] = config[key]
    return args
#%%
def main():
    global args
    args = vars(get_args(debug=False))
    args = load_config(args)
    wandb.config.update(args)
    
    save_dir = './assets/{}/{}_{}_sup{}/'.format(
        args["dataset"], args["labels"], args["prior"], str(args["sup_type"]))
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    
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

    args["cuda"] = torch.cuda.is_available()
    random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    if args["cuda"]:
        torch.cuda.manual_seed(args["seed"])
    global device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    global celoss
    celoss = torch.nn.BCEWithLogitsLoss()

    """dataset"""
    if args["dataset"] == "pendulum":
        class CustomDataset(Dataset): 
            def __init__(self, args):
                foldername = 'pendulum_real'
                self.name = ['light', 'angle', 'length', 'position', 'target']
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
        train_loader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)
    
    elif args["dataset"] == "celeba": 
        train_loader = None
        trans_f = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize((args["image_size"], args["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        data_dir = './modules/causal_data/celeba'
        if not os.path.exists(data_dir): 
            os.makedirs(data_dir)
        train_set = datasets.CelebA(data_dir, split='train', download=True, transform=trans_f)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args["batch_size"], 
                                                    shuffle=True, pin_memory=False,
                                                    drop_last=True, num_workers=0)

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

    print('Build models...')
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
    
    A_optimizer = None
    prior_optimizer = None
    if 'scm' in args["prior"]:
        enc_param = model.encoder.parameters()
        dec_param = list(model.decoder.parameters())
        prior_param = list(model.prior.parameters())
        
        A_optimizer = optim.Adam(prior_param[0:1], lr=args["lr_a"])
        prior_optimizer = optim.Adam(prior_param[1:], lr=args["lr_p"], betas=(args["beta1"], args["beta2"]))
    else:
        enc_param = model.encoder.parameters()
        dec_param = model.decoder.parameters()
        
    encoder_optimizer = optim.Adam(enc_param, lr=args["lr_e"], betas=(args["beta1"], args["beta2"]))
    decoder_optimizer = optim.Adam(dec_param, lr=args["lr_g"], betas=(args["beta1"], args["beta2"]))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=args["lr_d"], betas=(args["beta1"], args["beta2"]))

    model = model.to(device)
    discriminator = discriminator.to(device)

    # Fixed noise from prior p_z for generating from G
    global fixed_noise, fixed_unif_noise, fixed_zeros
    if args["prior"] == 'uniform':
        fixed_noise = torch.rand(args["save_n_samples"], args["latent_dim"], device=device) * 2 - 1
    else:
        fixed_noise = torch.randn(args["save_n_samples"], args["latent_dim"], device=device)
    fixed_unif_noise = torch.rand(1, args["latent_dim"], device=device) * 2 - 1
    fixed_zeros = torch.zeros(1, args["latent_dim"], device=device)

    # Train
    print('Start training...')
    for i in range(args["start_epoch"], args["start_epoch"] + args["n_epochs"]):
        train(i, model, discriminator, encoder_optimizer, decoder_optimizer, D_optimizer, train_loader, 
              label_idx, args["print_every"], save_dir, prior_optimizer, A_optimizer)
        # if i % args["save_model_every"] == 0:
        #     torch.save({'model': model.state_dict(), 
        #                 'discriminator': discriminator.state_dict()},
        #                 save_dir + 'model' + str(i) + '.sav')
    
    print('Model saving...')
    torch.save(model.state_dict(), save_dir + '/model_DEAR.pth')
    torch.save(discriminator.state_dict(), save_dir + '/discriminator_DEAR.pth')
    artifact = wandb.Artifact('model_DEAR', 
                            type='model', 
                            metadata=args) # description=""
    artifact.add_file(save_dir + '/model_DEAR.pth')
    artifact.add_file(save_dir + '/discriminator_DEAR.pth')
    artifact.add_file('./main.py')
    wandb.log_artifact(artifact)
    
    # """model load"""
    # artifact = wandb.use_artifact('anseunghwan/(causal)DEAR/model_{}:v{}'.format(args["dataset"], 0), type='model')
    # artifact.metadata
    # model_dir = artifact.download()
    # model_ = BGM(
    #     args["latent_dim"], 
    #     args["g_conv_dim"], 
    #     args["image_size"],
    #     args["enc_dist"], 
    #     args["enc_arch"], 
    #     args["enc_fc_size"], 
    #     args["enc_noise_dim"], 
    #     args["dec_dist"],
    #     args["prior"], 
    #     num_label, 
    #     A
    # )
    # discriminator_ = BigJointDiscriminator(
    #     args["latent_dim"], 
    #     args["d_conv_dim"], 
    #     args["image_size"],
    #     args["dis_fc_size"]
    # )
    # if args["cuda"]:
    #     model_.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format(args["dataset"])))
    #     discriminator_.load_state_dict(torch.load(model_dir + '/discriminator_{}.pth'.format(args["dataset"])))
    #     model_ = model_.to(device)
    #     discriminator_ = discriminator_.to(device)
    # else:
    #     model_.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format(args["dataset"]), 
    #                                       map_location=torch.device('cpu')))
    #     discriminator_.load_state_dict(torch.load(model_dir + '/discriminator_{}.pth'.format(args["dataset"]), 
    #                                               map_location=torch.device('cpu')))
    # x = torch.randn(10, 3, args["image_size"], args["image_size"]).to(device)
    # z = torch.randn(10, args["latent_dim"]).to(device)
    # out = model(x, z) 
    # out_ = model_(x, z)
    # out[-1] - out_[-1]
    # out = discriminator(x, z)
    # out_ = discriminator_(x, z)
    # out[0] - out_[0]
    
    wandb.run.finish()
#%%
# epoch, model, discriminator, encoder_optimizer, decoder_optimizer, D_optimizer, train_loader, label_idx, print_every, save_dir, prior_optimizer, A_optimizer = \
#     i, model, discriminator, encoder_optimizer, decoder_optimizer, D_optimizer, train_loader, label_idx, args["print_every"], save_dir, prior_optimizer, A_optimizer
def train(epoch, model, discriminator, encoder_optimizer, decoder_optimizer, D_optimizer, train_loader, 
          label_idx, print_every, save_dir, prior_optimizer, A_optimizer):
    
    model.train()
    discriminator.train()

    for batch_idx, (x, label) in enumerate(train_loader):
        x = x.to(device)
        
        # supervision flag
        sup_flag = label[:, 0] != -1
        if sup_flag.sum() > 0:
            label = label[sup_flag, :][:, label_idx].float()
        
        num_labels = len(label_idx)
        label = label.to(device)
        
        # with torch.autograd.set_detect_anomaly(True):

        """================== TRAIN DISCRIMINATOR =================="""
        for _ in range(args["d_steps_per_iter"]):
            discriminator.zero_grad()

            # Sample z from prior p_z
            if args["prior"] == 'uniform':
                z = torch.rand(x.size(0), args["latent_dim"], device=x.device) * 2 - 1
            else:
                z = torch.randn(x.size(0), args["latent_dim"], device=x.device)

            # Get inferred latent z = E(x) and generated image x = G(z)
            if 'scm' in args["prior"]:
                z_fake, x_fake, z, _ = model(x, z) # recon=False, infer_mean=True
            else:
                z_fake, x_fake, _ = model(x, z) # recon=False, infer_mean=True
            
            # Compute D loss
            encoder_score = discriminator(x, z_fake.detach()) # true (label = 0)
            decoder_score = discriminator(x_fake.detach(), z.detach()) # fake (label = 1)
            del z_fake
            del x_fake

            # softplus: log(1 + exp(x)), where x is logit
            # max (1 - label) * log(1 - sigmoid(decoder_score))
            #   + (label) * log(sigmoid(encoder_score))
            loss_d = F.softplus(decoder_score).mean() + F.softplus(-encoder_score).mean()
            
            loss_d.backward()
            D_optimizer.step()

        """================== TRAIN ENCODER & GENERATOR =================="""
        for _ in range(args["g_steps_per_iter"]):
            if args["prior"] == 'uniform':
                z = torch.rand(x.size(0), args["latent_dim"], device=x.device) * 2 - 1
            else:
                z = torch.randn(x.size(0), args["latent_dim"], device=x.device)
            
            if 'scm' in args["prior"]:
                z_fake, x_fake, z, z_fake_mean = model(x, z)
            else:
                z_fake, x_fake, z_fake_mean = model(x, z)

            """1. train encoder"""
            model.zero_grad()
            # WITH THE GENERATIVE LOSS
            encoder_score = discriminator(x, z_fake)
            loss_encoder = encoder_score.mean()

            # WITH THE SUPERVISED LOSS
            if sup_flag.sum() > 0:
                label_z = z_fake_mean[sup_flag, :num_labels]
                if 'pendulum' in args["dataset"]:
                    if args["sup_type"] == 'ce':
                        # CE loss
                        sup_loss = celoss(label_z, label)
                    else:
                        # l2 loss
                        sup_loss = nn.MSELoss()(label_z, label)
                else:
                    sup_loss = celoss(label_z, label)
            else:
                sup_loss = torch.zeros([1], device=device)
            loss_encoder = loss_encoder + sup_loss * args["sup_coef"]

            """2. train generator"""
            decoder_score = discriminator(x_fake, z)
            # with scaling clipping for stabilization
            r_decoder = torch.exp(decoder_score.detach())
            s_decoder = r_decoder.clamp(0.5, 2)
            loss_decoder = -(s_decoder * decoder_score).mean()
            
            loss = loss_encoder + loss_decoder
            loss.backward()
            
            encoder_optimizer.step()
            decoder_optimizer.step()
            # if 'scm' in args["prior"]:
            #     prior_optimizer.step()
            
            if 'scm' in args["prior"]:
                model.prior.set_zero_grad()
                A_optimizer.step()
                prior_optimizer.step()
            
        # Print out losses
        if batch_idx == 0 or (batch_idx + 1) % print_every == 0:
            log = ('Train Epoch: {} ({:.0f}%)\tD loss: {:.4f}, Encoder loss: {:.4f}, Decoder loss: {:.4f}, Sup loss: {:.4f}, '
                   'E_score: {:.4f}, D score: {:.4f}'.format(
                epoch, 100. * batch_idx / len(train_loader),
                loss_d.item(), loss_encoder.item(), loss_decoder.item(), sup_loss.item(),
                encoder_score.mean().item(), decoder_score.mean().item()))
            print(log)
            wandb.log({"loss(Discriminator)": loss_d.item(),
                       "loss(Encoder)": loss_encoder.item(),
                       "loss(Decoder)": loss_decoder.item(),
                       "loss(Sup)": sup_loss.item()})

        if (epoch == 1 or epoch % args["sample_every_epoch"] == 0) and batch_idx == len(train_loader) - 1:
            test(epoch, batch_idx + 1, model, x[:args["save_n_recons"]], save_dir)

            """estimated causal adjacency matrix"""
            B_est = model.prior.A.detach().cpu().numpy()
            fig = viz_heatmap(np.flipud(B_est), size=(7, 7))
            wandb.log({'B_est': wandb.Image(fig)})
#%%
def draw_recon(x, x_recon):
    x_l, x_recon_l = x.tolist(), x_recon.tolist()
    result = [None] * (len(x_l) + len(x_recon_l))
    result[::2] = x_l
    result[1::2] = x_recon_l
    return torch.FloatTensor(result)
#%%
# epoch, i, model, test_data, save_dir = epoch, batch_idx + 1, model, x[:args["save_n_recons"]], save_dir
def test(epoch, i, model, test_data, save_dir):
    model.eval()
    with torch.no_grad():
        x = test_data.to(device)

        # Reconstruction
        x_recon = model(x, recon=True)
        recons = draw_recon(x.cpu(), x_recon.cpu())
        save_image(recons, save_dir + 'recon_' + str(epoch) + '_' + str(i) + '.png', nrow=args["nrow"],
                   normalize=True, scale_each=True)
        fig = Image.open(save_dir + 'recon_' + str(epoch) + '_' + str(i) + '.png')
        wandb.log({'reconstruction': wandb.Image(fig)})

        # Generation
        sample = model(z=fixed_noise).cpu()
        save_image(sample, save_dir + 'gen_' + str(epoch) + '_' + str(i) + '.png', 
                   normalize=True, scale_each=True)
        fig = Image.open(save_dir + 'gen_' + str(epoch) + '_' + str(i) + '.png')
        wandb.log({'generation': wandb.Image(fig)})

        # Traversal (given a fixed traversal range)
        sample = model.traverse(fixed_zeros).cpu()
        save_image(sample, save_dir + 'trav_' + str(epoch) + '_' + str(i) + '.png', 
                   normalize=True, scale_each=True, nrow=10)
        fig = Image.open(save_dir + 'trav_' + str(epoch) + '_' + str(i) + '.png')
        wandb.log({'traversal': wandb.Image(fig)})
        del sample

    model.train()
#%%
def get_scale():
    '''return max and min of training data'''
    scale = torch.Tensor([[0.0000, 48.0000, 2.0000, 2.0178], 
                          [40.5000, 88.5000, 14.8639, 14.4211]])
    return scale
#%%
def get_stats():
    '''return mean and std of training data'''
    mm = torch.Tensor([20.2500, 68.2500, 6.9928, 8.7982])
    ss = torch.Tensor([11.8357, 11.8357, 2.8422, 2.1776])
    return mm, ss
#%%
if __name__ == '__main__':
    main()
#%%