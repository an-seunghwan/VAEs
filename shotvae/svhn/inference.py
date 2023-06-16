#%%
import argparse
import os

# os.chdir(r'D:\semi\shotvae') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/shotvae') # main directory (repository)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import tqdm
import yaml
import io
import matplotlib.pyplot as plt
import random as python_random

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

from preprocess import fetch_dataset
from model import VAE
from criterion import ELBO_criterion
from mixup import augment, optimal_match_mix, weight_decay_decoupled, label_smoothing 
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

run = wandb.init(
    project="EXoN", 
    entity="anseunghwan",
    tags=["svhn", "shotvae", "inference"],
)
#%%
import ast
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v
#%%
def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--dataset', type=str, default='svhn',
                        help='dataset used for training')
    parser.add_argument('--num', type=int, default=5, 
                        help='seed for repeatable results')

    return parser
#%%
def main():
    #%%
    # '''argparse to dictionary'''
    # args = vars(get_args())
    '''argparse debugging'''
    args = vars(get_args().parse_args(args=[]))
    #%%
    """model load"""
    artifact = wandb.use_artifact(f'anseunghwan/EXoN/{args["dataset"]}_PartedVAE:v{args["num"]}', type='model')
    # artifact = wandb.use_artifact(f'anseunghwan/EXoN/{args["dataset"]}_ShotVAE:v{args["num"]}', type='model')
    for key, item in artifact.metadata.items():
        args[key] = item
    model_dir = artifact.download()
    #%%
    np.random.seed(args["seed"])
    python_random.seed(args["seed"])
    tf.random.set_seed(args["seed"])
    #%%
    num_classes = 10
    model_name = [x for x in os.listdir(model_dir) if x.endswith('h5')][0]
    model = VAE(num_classes=num_classes, depth=args['depth'], width=args['width'], slope=args['slope'],
                latent_dim=args['ldc'], temperature=args['temperature'])
    model.build(input_shape=[(None, 32, 32, 3), (None, num_classes)])
    model.load_weights(model_dir + '/' + model_name)
    model.summary()
    #%%
    z_list = []
    idx_list = [5, 20]
    img_list = []
    for idx in idx_list:
        img = (np.load(f"./assets/report/img{idx}.npy") + 1) / 2
        label = np.load(f"./assets/report/label{idx}.npy")
        mean, log_sigma, log_prob, z, y, xhat = model([img[tf.newaxis, ...], label[tf.newaxis, ...]], training=False)
        z_list.append(mean)
        img_list.append(img)
    
    z_inter = np.linspace(z_list[0][0], z_list[1][0], 8)
    label = tf.repeat(label[tf.newaxis, ...], repeats=len(z_inter), axis=0)
    inter_recon = model.Decoder(tf.concat([z_inter, label], axis=-1), training=False)
    
    fig, axes = plt.subplots(1, 10, figsize=(25, 5))
    axes.flatten()[0].imshow(img_list[0])
    axes.flatten()[0].axis('off')
    for i in range(8):
        axes.flatten()[i+1].imshow(inter_recon[i].numpy())
        axes.flatten()[i+1].axis('off')
    axes.flatten()[9].imshow(img_list[1])
    axes.flatten()[9].axis('off')
    plt.tight_layout()
    plt.savefig(f'./assets/shotvae_svhn_inter_same.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    wandb.log({f'between same': wandb.Image(fig)})
    #%%
    wandb.config.update(args, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%