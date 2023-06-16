#%%
import argparse
import os

# os.chdir(r'D:\semi\dgm') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/dgm') # main directory (repository)
# os.chdir('/Users/anseunghwan/Documents/GitHub/semi/dgm')
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
from model import DGM
from criterion import ELBO_criterion
from utils import weight_decay_decoupled
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
    tags=["svhn", "M2", "inference"],
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
    artifact = wandb.use_artifact(f'anseunghwan/EXoN/{args["dataset"]}_M2:v{args["num"]}', type='model')
    for key, item in artifact.metadata.items():
        args[key] = item
    model_dir = artifact.download()
    #%%
    np.random.seed(args["seed"])
    python_random.seed(args["seed"])
    tf.random.set_seed(args["seed"])
    #%%
    save_path = os.path.dirname(os.path.abspath(__file__)) + '/dataset/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args['dataset'], save_path, args)
    total_length = sum(1 for _ in datasetU)
    #%%
    model_name = [x for x in os.listdir(model_dir) if x.endswith('h5')][0]
    model = DGM(num_classes,
                latent_dim=args['latent_dim'],
                dropratio=args['drop_rate'])
    model.classifier.build(input_shape=(None, 32, 32, 3))
    model.build(input_shape=[(None, 32, 32, 3), (None, num_classes)])
    model.load_weights(model_dir + '/' + model_name)
    model.summary()
    #%%
    autotune = tf.data.AUTOTUNE
    shuffle_and_batchL = lambda dataset: dataset.batch(
        batch_size=args['labeled_batch_size'], 
        drop_remainder=True).prefetch(autotune)
    iteratorL = iter(shuffle_and_batchL(datasetL))
    imageL, labelL = next(iteratorL)
    #%%
    for idx in [6, 31]:
        mean, logvar, z = model.encode([imageL[idx][tf.newaxis, ...], labelL[idx][tf.newaxis, ...]], training=False)
        
        figure = plt.figure(figsize=(10, 2))
        plt.subplot(1, num_classes+1, 1)
        plt.imshow(imageL[idx])
        plt.axis('off')
        for i in range(num_classes):
            label = np.zeros((z.shape[0], num_classes))
            label[:, i] = 1
            xhat = model.decode(mean, label, training=False)
            plt.subplot(1, num_classes+1, i+2)
            plt.imshow(xhat[0])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'./assets/M2_svhn_recon_{idx}.png', bbox_inches='tight')
        plt.show()
        plt.close()
        
        wandb.log({f'recon ({idx})': wandb.Image(figure)})
    #%%
    wandb.config.update(args, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%