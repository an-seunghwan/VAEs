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
    tags=["svhn", "M2"],
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
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--labeled-batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size for labeled dataset (default: 32)')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=200, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct_freq', '-rf', default=10, type=int,
                        metavar='N', help='reconstruct frequency (default: 10)')
    parser.add_argument('--labeled_examples', type=int, default=1000, 
                        help='number labeled examples (default: 1000), all labels are balanced')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')

    '''Deep VAE Model Parameters'''
    parser.add_argument("--bce_reconstruction", default=True, type=bool,
                        help="Do BCE Reconstruction")
    parser.add_argument('--drop_rate', default=0.1, type=float, 
                        help='drop rate for the network')

    '''VAE parameters'''
    parser.add_argument('--latent_dim', default=128, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    
    '''Optimizer Parameters'''
    parser.add_argument('--learning_rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--alpha', default=1, type=float,
                        help='weight of supervised classification loss')

    '''Configuration'''
    parser.add_argument('--config_path', type=str, default=None, 
                        help='path to yaml config file, overwrites args')

    return parser.parse_args()
#%%
def load_config(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(dir_path, args['config_path'])    
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in args.keys():
        if key in config.keys():
            args[key] = config[key]
    return args
#%%
def generate_and_save_images(image, num_classes, step, save_dir):
    fig = plt.figure(figsize=(10, 2))
    for i in range(num_classes):
        plt.subplot(1, num_classes, i+1)
        plt.imshow(image[i])
        plt.axis('off')
    plt.savefig('{}/imgs/image_at_epoch_{}.png'.format(save_dir, step))
    # plt.show()
    plt.close()
    return fig
#%%
def main():
    #%%
    '''argparse to dictionary'''
    args = vars(get_args())
    # '''argparse debugging'''
    # args = vars(parser.parse_args(args=[]))
    wandb.config.update(args)
    #%%
    np.random.seed(args["seed"])
    python_random.seed(args["seed"])
    tf.random.set_seed(args["seed"])
    #%%
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
        args = load_config(args)

    model_path = f'./assets/{current_time}'
    if not os.path.exists(f'{model_path}'):
        os.makedirs(f'{model_path}')
    if not os.path.exists(f'{model_path}/imgs'):
        os.makedirs(f'{model_path}/imgs')
    
    save_path = os.path.dirname(os.path.abspath(__file__)) + '/dataset/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args['dataset'], save_path, args)
    total_length = sum(1 for _ in datasetU)
    #%%
    model = DGM(num_classes,
                latent_dim=args['latent_dim'],
                dropratio=args['drop_rate'])
    model.classifier.build(input_shape=(None, 32, 32, 3))
    model.build(input_shape=[(None, 32, 32, 3), (None, num_classes)])
    model.summary()
    
    buffer_model = DGM(num_classes,
                    latent_dim=args['latent_dim'],
                    dropratio=args['drop_rate'])
    buffer_model.classifier.build(input_shape=(None, 32, 32, 3))
    buffer_model.build(input_shape=[(None, 32, 32, 3), (None, num_classes)])
    buffer_model.set_weights(model.get_weights()) # weight initialization
    #%%
    '''optimizer'''
    optimizer = K.optimizers.Adam(learning_rate=args['learning_rate'])
    optimizer_classifier = K.optimizers.Adam(learning_rate=args['learning_rate'] * 0.1)

    test_accuracy_print = 0.
    
    '''weight of KL-divergence'''
    beta = tf.cast(1, tf.float32) 
    #%%
    for epoch in range(args['start_epoch'], args['epochs']):
        
        '''learning rate schedule'''
        lr_gamma = 0.5
        if epoch % 5 == 0 and epoch != 0:
            optimizer_classifier.lr = optimizer_classifier.lr * lr_gamma
            
        if epoch % args['reconstruct_freq'] == 0:
            loss, recon_loss, elboL_loss, elboU_loss, kl_loss, accuracy, sample_recon = train(
                datasetL, datasetU, model, buffer_model, optimizer, optimizer_classifier, epoch, args, beta, num_classes, total_length, test_accuracy_print, model_path
            )
            wandb.log({'(train) sample_recon': wandb.Image(sample_recon)})
        else:
            loss, recon_loss, elboL_loss, elboU_loss, kl_loss, accuracy = train(
                datasetL, datasetU, model, buffer_model, optimizer, optimizer_classifier, epoch, args, beta, num_classes, total_length, test_accuracy_print, model_path
            )
        val_recon_loss, val_kl_loss, val_elbo_loss, val_accuracy = validate(
            val_dataset, model, epoch, beta, args, num_classes, split='Validation'
        )
        test_recon_loss, test_kl_loss, test_elbo_loss, test_accuracy = validate(
            test_dataset, model, epoch, beta, args, num_classes, split='Test'
        )
        
        wandb.log({'(train) loss': loss.result().numpy()})
        wandb.log({'(train) recon_loss': recon_loss.result().numpy()})
        wandb.log({'(train) elboL_loss': elboL_loss.result().numpy()})
        wandb.log({'(train) elboU_loss': elboU_loss.result().numpy()})
        wandb.log({'(train) kl_loss': kl_loss.result().numpy()})
        wandb.log({'(train) accuracy': accuracy.result().numpy()})
        
        wandb.log({'(val) recon_loss': val_recon_loss.result().numpy()})
        wandb.log({'(val) kl_loss': val_kl_loss.result().numpy()})
        wandb.log({'(val) elbo_loss': val_elbo_loss.result().numpy()})
        wandb.log({'(val) accuracy': val_accuracy.result().numpy()})
        
        wandb.log({'(test) recon_loss': test_recon_loss.result().numpy()})
        wandb.log({'(test) kl_loss': test_kl_loss.result().numpy()})
        wandb.log({'(test) elbo_loss': test_elbo_loss.result().numpy()})
        wandb.log({'(test) accuracy': test_accuracy.result().numpy()})
        
        test_accuracy_print = test_accuracy.result()

        # Reset metrics every epoch
        loss.reset_states()
        recon_loss.reset_states()
        kl_loss.reset_states()
        elboL_loss.reset_states()
        elboU_loss.reset_states()
        accuracy.reset_states()
        val_recon_loss.reset_states()
        val_kl_loss.reset_states()
        val_elbo_loss.reset_states()
        val_accuracy.reset_states()
        test_recon_loss.reset_states()
        test_kl_loss.reset_states()
        test_elbo_loss.reset_states()
        test_accuracy.reset_states()
    #%%
    '''model & configurations save'''        
    model.save_weights(model_path + '/model.h5', save_format="h5")

    with open(model_path + '/args.txt', "w") as f:
        for key, value, in args.items():
            f.write(str(key) + ' : ' + str(value) + '\n')
    
    artifact = wandb.Artifact(
        f'{args["dataset"]}_M2', 
        type='model',
        metadata=args) # description=""
    artifact.add_file(model_path + '/args.txt')
    artifact.add_file(model_path + '/model.h5')
    artifact.add_file('./main.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(args, allow_val_change=True)
    wandb.run.finish()
#%%
def train(datasetL, datasetU, model, buffer_model, optimizer, optimizer_classifier, epoch, args, beta, num_classes, total_length, test_accuracy_print, model_path):
    loss_avg = tf.keras.metrics.Mean()
    recon_loss_avg = tf.keras.metrics.Mean()
    elboL_loss_avg = tf.keras.metrics.Mean()
    elboU_loss_avg = tf.keras.metrics.Mean()
    kl_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    '''supervised classification weight'''
    alpha = tf.cast(args['alpha'] * total_length / args['labeled_examples'], tf.float32)
    
    autotune = tf.data.AUTOTUNE
    shuffle_and_batchL = lambda dataset: dataset.shuffle(buffer_size=int(1e3)).batch(batch_size=args['labeled_batch_size'], 
                                                                                    drop_remainder=True).prefetch(autotune)
    shuffle_and_batchU = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'], 
                                                                                    drop_remainder=True).prefetch(autotune)

    iteratorL = iter(shuffle_and_batchL(datasetL))
    iteratorU = iter(shuffle_and_batchU(datasetU))
        
    iteration = total_length // args['batch_size'] 
    
    progress_bar = tqdm.tqdm(range(iteration), unit='batch')
    for batch_num in progress_bar:
        
        try:
            imageL, labelL = next(iteratorL)
        except:
            iteratorL = iter(shuffle_and_batchL(datasetL))
            imageL, labelL = next(iteratorL)
        try:
            imageU, _ = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batchU(datasetU))
            imageU, _ = next(iteratorU)
        
        with tf.GradientTape(persistent=True) as tape:    
            '''labeled'''
            mean, logvar, z, xhat = model([imageL, labelL])
            recon_loss, prior_y, pz, qz = ELBO_criterion(xhat, imageL, labelL, z, mean, logvar, num_classes, args)
            elboL = tf.reduce_mean(recon_loss - prior_y + beta * (qz - pz))
            
            '''unlabeled'''
            with tape.stop_recording():
                labelU = tf.concat([tf.one_hot(i, depth=num_classes)[tf.newaxis, ] for i in range(num_classes)], axis=0)[tf.newaxis, ...]
                labelU = tf.repeat(labelU, tf.shape(imageU)[0], axis=0)
                labelU = tf.reshape(labelU, (-1, num_classes))
                
            imageU_ = imageU[:, tf.newaxis, :, :, :]
            imageU_ = tf.reshape(tf.repeat(imageU_, num_classes, axis=1), (-1, 32, 32, 3))
            
            mean, logvar, z, xhat = model([imageU_, labelU])
            recon_loss, prior_y, pz, qz = ELBO_criterion(xhat, imageU_, labelU, z, mean, logvar, num_classes, args)
            
            recon_loss = tf.reshape(recon_loss, (tf.shape(imageU)[0], num_classes, -1))
            prior_y = tf.reshape(prior_y, (tf.shape(imageU)[0], num_classes, -1))
            pz = tf.reshape(pz, (tf.shape(imageU)[0], num_classes, -1))
            qz = tf.reshape(qz, (tf.shape(imageU)[0], num_classes, -1))
            
            probU = model.classify(imageU)
            elboU = recon_loss - prior_y + beta * (qz - pz)
            elboU = tf.reduce_mean(tf.reduce_sum(probU[..., tf.newaxis] * elboU, axis=[1, 2]))
            entropyU = - tf.reduce_mean(tf.reduce_sum(probU * tf.math.log(tf.clip_by_value(probU, 1e-6, 1.)), axis=-1))
            elboU -= entropyU
            
            '''supervised classification loss'''
            probL = model.classify(imageL)
            cce = - tf.reduce_mean(tf.reduce_sum(tf.multiply(labelL, tf.math.log(tf.clip_by_value(probL, 1e-6, 1.))), axis=-1))
            
            loss = elboL + elboU + alpha * cce
            
        grads = tape.gradient(loss, model.encoder.trainable_variables + model.decoder.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.encoder.trainable_variables + model.decoder.trainable_variables))
        # classifier
        grads = tape.gradient(loss, model.classifier.trainable_variables) 
        optimizer_classifier.apply_gradients(zip(grads, model.classifier.trainable_variables)) 
        '''decoupled weight decay'''
        weight_decay_decoupled(model.classifier, buffer_model.classifier, decay_rate=args['weight_decay'] * optimizer_classifier.lr)
        
        loss_avg(loss)
        elboL_loss_avg(elboL)
        elboU_loss_avg(elboU)
        recon_loss_avg(recon_loss)
        kl_loss_avg(qz - pz)
        accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), probL)
        
        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'Loss': f'{loss_avg.result():.4f}',
            'ELBO(L)': f'{elboL_loss_avg.result():.4f}',
            'ELBO(U)': f'{elboU_loss_avg.result():.4f}',
            'Recon': f'{recon_loss_avg.result():.4f}',
            'KL': f'{kl_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}',
            'Test Accuracy': f'{test_accuracy_print:.3%}',
            'beta': f'{beta:.4f}'
        })
    
    if epoch % args['reconstruct_freq'] == 0:
        sample_recon = generate_and_save_images(xhat, num_classes, epoch, model_path)
        return loss_avg, recon_loss_avg, elboL_loss_avg, elboU_loss_avg, kl_loss_avg, accuracy, sample_recon
    else:
        return loss_avg, recon_loss_avg, elboL_loss_avg, elboU_loss_avg, kl_loss_avg, accuracy
#%%
def validate(dataset, model, epoch, beta, args, num_classes, split):
    recon_loss_avg = tf.keras.metrics.Mean()   
    kl_loss_avg = tf.keras.metrics.Mean()   
    elbo_loss_avg = tf.keras.metrics.Mean()   
    accuracy = tf.keras.metrics.Accuracy()
    
    dataset = dataset.batch(args['batch_size'], drop_remainder=False)
    for image, label in dataset:
        mean, logvar, z, xhat = model([image, label], training=False)
        recon_loss, prior_y, pz, qz = ELBO_criterion(xhat, image, label, z, mean, logvar, num_classes, args)
        elbo = tf.reduce_mean(recon_loss - prior_y + beta * (qz - pz))
        prob = model.classify(image, training=False)
        
        recon_loss_avg(recon_loss)
        kl_loss_avg(qz - pz)
        elbo_loss_avg(elbo)
        accuracy(tf.argmax(prob, axis=1, output_type=tf.int32), 
                 tf.argmax(label, axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split} ELBO Loss: {elbo_loss_avg.result():.4f}, Recon: {recon_loss_avg.result():.4f}, KL: {kl_loss_avg.result():.4f}, Accuracy: {accuracy.result():.3%}')
    
    return recon_loss_avg, kl_loss_avg, elbo_loss_avg, accuracy
#%%
if __name__ == '__main__':
    main()
#%%