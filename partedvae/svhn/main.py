#%%
import argparse
import os

# os.chdir(r'D:\semi\partedvae') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/partedvae') # main directory (repository)
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
from model import VAE
from criterion import ELBO_criterion
from utils import CustomReduceLRoP
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
    tags=["svhn", "partedvae"],
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

    '''VAE parameters'''
    parser.add_argument('--z_dim', default=64, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    parser.add_argument('--u_dim', default=64, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    parser.add_argument('--depth', type=int, default=28, 
                        help='depth for WideResnet (default: 28)')
    parser.add_argument('--width', type=int, default=4, 
                        help='widen factor for WideResnet (default: 2)')
    parser.add_argument('--slope', type=float, default=0.1, 
                        help='slope parameter for LeakyReLU (default: 0.1)')
    
    '''Optimizer Parameters'''
    parser.add_argument('--learning_rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--classifier_learning_rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate for classifier')
    
    parser.add_argument("--z_capacity", default=[0., 30., 150000, 1000.], type=arg_as_list,
                        help="controlled capacity") # cap_min, cap_max, num_iters, gamma
    parser.add_argument("--u_capacity", default=[0., 30., 150000, 1000.], type=arg_as_list,
                        help="controlled capacity") # cap_min, cap_max, num_iters, gamma
    parser.add_argument('--gamma_c', default=1000, type=float,
                        help='weight of loss')
    parser.add_argument('--gamma_h', default=2000, type=float,
                        help='weight of loss')
    parser.add_argument('--gamma_bc', default=2000, type=float,
                        help='weight of loss')
    parser.add_argument('--bc_threshold', default=0.2, type=float,
                        help='threshold of Bhattacharyya coefficient')

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
    iteration = total_length // args['batch_size'] 
    #%%
    model = VAE(
        num_classes=num_classes,
        latent_dim=args['z_dim'], 
        u_dim=args['u_dim'],
        depth=args['depth'], width=args['width'], slope=args['slope']
    )
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    
    '''optimizer'''
    optimizer = K.optimizers.Adam(learning_rate=args['learning_rate'])
    optimizer_classifier = K.optimizers.Adam(learning_rate=args['classifier_learning_rate'])

    '''used in computing BC'''
    BC_valid_mask = np.ones((num_classes, num_classes))
    BC_valid_mask = tf.constant(np.tril(BC_valid_mask, k=-1), tf.float32)

    test_accuracy_print = 0.
    #%%
    for epoch in range(args['start_epoch'], args['epochs']):
        
        '''learning rate schedule'''
        optimizer_classifier_scheduler = CustomReduceLRoP(
            factor=0.5,
            patience=2,
            min_delta=1e-1,
            mode='auto',
            cooldown=3,
            min_lr=0,
            optim_lr=optimizer_classifier.learning_rate, 
            reduce_lin=True,
            verbose=1
        )
        optimizer_scheduler = CustomReduceLRoP(
            factor=0.5,
            patience=2,
            min_delta=1e-2,
            mode='auto',
            cooldown=4,
            min_lr=0,
            optim_lr=optimizer.learning_rate, 
            reduce_lin=True,
            verbose=1
        )

        if epoch % args['reconstruct_freq'] == 0:
            ce_loss, loss, recon_loss, z_loss, c_loss, c_entropy_loss, u_loss, prior_intersection_loss, accuracy, sample_recon = train(
                datasetL, datasetU, model, optimizer, optimizer_classifier, epoch, BC_valid_mask, args, num_classes, iteration, test_accuracy_print, model_path
            )
            wandb.log({'(train) sample_recon': wandb.Image(sample_recon)})
        else:
            ce_loss, loss, recon_loss, z_loss, c_loss, c_entropy_loss, u_loss, prior_intersection_loss, accuracy = train(
                datasetL, datasetU, model, optimizer, optimizer_classifier, epoch, BC_valid_mask, args, num_classes, iteration, test_accuracy_print, model_path
            )
        val_loss, val_recon_loss, val_z_loss, val_c_loss, val_c_entropy_loss, val_u_loss, val_prior_intersection_loss, val_accuracy = validate(
            val_dataset, model, epoch, iteration, iteration, BC_valid_mask, args, num_classes, split='Validation'
        )
        test_loss, test_recon_loss, test_z_loss, test_c_loss, test_c_entropy_loss, test_u_loss, test_prior_intersection_loss, test_accuracy = validate(
            test_dataset, model, epoch, iteration, iteration, BC_valid_mask, args, num_classes, split='Test'
        )
        
        wandb.log({'(train) loss': loss.result().numpy()})
        wandb.log({'(train) ce_loss': ce_loss.result().numpy()})
        wandb.log({'(train) recon_loss': recon_loss.result().numpy()})
        wandb.log({'(train) z_loss': z_loss.result().numpy()})
        wandb.log({'(train) c_loss': c_loss.result().numpy()})
        wandb.log({'(train) c_entropy_loss': c_entropy_loss.result().numpy()})
        wandb.log({'(train) u_loss': u_loss.result().numpy()})
        wandb.log({'(train) prior_intersection_loss': prior_intersection_loss.result().numpy()})
        wandb.log({'(train) accuracy': accuracy.result().numpy()})
        
        wandb.log({'(val) loss': val_loss.result().numpy()})
        wandb.log({'(val) recon_loss': val_recon_loss.result().numpy()})
        wandb.log({'(val) z_loss': val_z_loss.result().numpy()})
        wandb.log({'(val) c_loss': val_c_loss.result().numpy()})
        wandb.log({'(val) c_entropy_loss': val_c_entropy_loss.result().numpy()})
        wandb.log({'(val) u_loss': val_u_loss.result().numpy()})
        wandb.log({'(val) prior_intersection_loss': val_prior_intersection_loss.result().numpy()})
        wandb.log({'(val) accuracy': val_accuracy.result().numpy()})
        
        wandb.log({'(test) loss': test_loss.result().numpy()})
        wandb.log({'(test) recon_loss': test_recon_loss.result().numpy()})
        wandb.log({'(test) z_loss': test_z_loss.result().numpy()})
        wandb.log({'(test) c_loss': test_c_loss.result().numpy()})
        wandb.log({'(test) c_entropy_loss': test_c_entropy_loss.result().numpy()})
        wandb.log({'(test) u_loss': test_u_loss.result().numpy()})
        wandb.log({'(test) prior_intersection_loss': test_prior_intersection_loss.result().numpy()})
        wandb.log({'(test) accuracy': test_accuracy.result().numpy()})
            
        test_accuracy_print = test_accuracy.result()
        
        '''optimizer scheduling'''
        optimizer_classifier_scheduler.on_epoch_end(epoch, ce_loss.result())
        optimizer_scheduler.on_epoch_end(epoch, loss.result())

        # Reset metrics every epoch
        loss.reset_states()
        recon_loss.reset_states()
        z_loss.reset_states()
        c_loss.reset_states()
        c_entropy_loss.reset_states()
        u_loss.reset_states()
        prior_intersection_loss.reset_states()
        accuracy.reset_states()
        val_loss.reset_states()
        val_recon_loss.reset_states()
        val_z_loss.reset_states()
        val_c_loss.reset_states()
        val_c_entropy_loss.reset_states()
        val_u_loss.reset_states()
        val_prior_intersection_loss.reset_states()
        val_accuracy.reset_states()
        test_loss.reset_states()
        test_recon_loss.reset_states()
        test_z_loss.reset_states()
        test_c_loss.reset_states()
        test_c_entropy_loss.reset_states()
        test_u_loss.reset_states()
        test_prior_intersection_loss.reset_states()
        test_accuracy.reset_states()
    #%%
    '''model & configurations save'''        
    model.save_weights(model_path + '/model.h5', save_format="h5")

    with open(model_path + '/args.txt', "w") as f:
        for key, value, in args.items():
            f.write(str(key) + ' : ' + str(value) + '\n')
    
    artifact = wandb.Artifact(
        f'{args["dataset"]}_PartedVAE', 
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
def train(datasetL, datasetU, model, optimizer, optimizer_classifier, epoch, BC_valid_mask, args, num_classes, iteration, test_accuracy_print, model_path):
    ce_loss_avg = tf.keras.metrics.Mean()
    loss_avg = tf.keras.metrics.Mean()
    recon_loss_avg = tf.keras.metrics.Mean()
    z_loss_avg = tf.keras.metrics.Mean()
    c_loss_avg = tf.keras.metrics.Mean()
    c_entropy_loss_avg = tf.keras.metrics.Mean()
    u_loss_avg = tf.keras.metrics.Mean()
    prior_intersection_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    autotune = tf.data.AUTOTUNE
    shuffle_and_batchL = lambda dataset: dataset.shuffle(buffer_size=int(1e3)).batch(batch_size=args['labeled_batch_size'], 
                                                                                    drop_remainder=True).prefetch(autotune)
    shuffle_and_batchU = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'] - args['labeled_batch_size'], 
                                                                                    drop_remainder=True).prefetch(autotune)

    iteratorL = iter(shuffle_and_batchL(datasetL))
    iteratorU = iter(shuffle_and_batchU(datasetU))
        
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
        
        image = tf.concat([imageL, imageU], axis=0)
        
        '''1. classifier training (warm-up)'''
        with tf.GradientTape(persistent=True) as tape:    
            prob = model.classify(imageL)
            prob = tf.clip_by_value(prob, 1e-10, 1.)
            ce_loss = tf.reduce_mean(- tf.reduce_sum(labelL * tf.math.log(prob), axis=-1))
            
        grads = tape.gradient(ce_loss, model.feature_extractor.trainable_variables + model.h_to_c_logit.trainable_variables) 
        optimizer_classifier.apply_gradients(zip(grads, model.feature_extractor.trainable_variables + model.h_to_c_logit.trainable_variables)) 
            
        '''2. objective training'''
        with tf.GradientTape(persistent=True) as tape:    
            z_mean, z_logvar, z, c_logit, u_mean, u_logvar, u, xhat = model(image)
            
            recon_loss, z_kl, agg_c_kl, c_entropy, u_kl, BC = ELBO_criterion(
                xhat, image, z_mean, z_logvar, c_logit, u_mean, u_logvar, model, num_classes, args
            )
            
            '''KL-divergence of z'''
            cap_min, cap_max, num_iters, gamma = args['z_capacity']
            num_steps = epoch * iteration + batch_num
            cap_current = (cap_max - cap_min) * (num_steps / num_iters) + cap_min
            cap_current = tf.math.minimum(cap_current, cap_max)
            z_loss = gamma * tf.math.abs(z_kl - cap_current)
            
            '''KL-divergence of c (marginal)'''
            c_loss = args['gamma_c'] * agg_c_kl
            
            '''entropy of c'''
            c_entropy_loss = args['gamma_h'] * c_entropy
            
            '''mixture KL-divergence of u'''
            cap_min, cap_max, num_iters, gamma = args['u_capacity']
            num_steps = epoch * iteration + batch_num
            cap_current = (cap_max - cap_min) * (num_steps / num_iters) + cap_min
            cap_current = tf.math.minimum(cap_current, cap_max)
            u_loss = gamma * tf.math.abs(u_kl - cap_current)
            
            '''Bhattacharyya coefficient'''
            valid_BC = BC * BC_valid_mask
            valid_BC = tf.clip_by_value(valid_BC - args['bc_threshold'], 0., 1.)
            prior_intersection_loss = args['gamma_bc'] * tf.reduce_sum(valid_BC)
            
            loss = recon_loss + z_loss + c_loss + c_entropy_loss + u_loss + prior_intersection_loss
            
        grads = tape.gradient(loss, model.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        ce_loss_avg(ce_loss)    
        loss_avg(loss)
        recon_loss_avg(recon_loss)
        z_loss_avg(z_loss)
        c_loss_avg(c_loss)
        c_entropy_loss_avg(c_entropy_loss)
        u_loss_avg(u_loss)
        prior_intersection_loss_avg(prior_intersection_loss)
        accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), prob)
        
        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'Loss': f'{loss_avg.result():.4f}',
            'Recon': f'{recon_loss_avg.result():.4f}',
            'Z_Loss': f'{z_loss_avg.result():.4f}',
            'C_Loss': f'{c_loss_avg.result():.4f}',
            'C_Entropy': f'{c_entropy_loss_avg.result():.4f}',
            'U_Loss': f'{u_loss_avg.result():.4f}',
            'Prior_intersection': f'{prior_intersection_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}',
            'Test Accuracy': f'{test_accuracy_print:.3%}',
        })
    
    if epoch % args['reconstruct_freq'] == 0:
        sample_recon = generate_and_save_images(xhat, num_classes, epoch, model_path)
        return ce_loss_avg, loss_avg, recon_loss_avg, z_loss_avg, c_loss_avg, c_entropy_loss_avg, u_loss_avg, prior_intersection_loss_avg, accuracy, sample_recon
    else:
        return ce_loss_avg, loss_avg, recon_loss_avg, z_loss_avg, c_loss_avg, c_entropy_loss_avg, u_loss_avg, prior_intersection_loss_avg, accuracy
#%%
def validate(dataset, model, epoch, iteration, batch_num, BC_valid_mask, args, num_classes, split):
    loss_avg = tf.keras.metrics.Mean()
    recon_loss_avg = tf.keras.metrics.Mean()
    z_loss_avg = tf.keras.metrics.Mean()
    c_loss_avg = tf.keras.metrics.Mean()
    c_entropy_loss_avg = tf.keras.metrics.Mean()
    u_loss_avg = tf.keras.metrics.Mean()
    prior_intersection_loss_avg = tf.keras.metrics.Mean()  
    accuracy = tf.keras.metrics.Accuracy()
    
    dataset = dataset.batch(args['batch_size'], drop_remainder=False)
    for image, label in dataset:
        z_mean, z_logvar, z, c_logit, u_mean, u_logvar, u, xhat = model(image, training=False)
        prob = tf.nn.softmax(c_logit, axis=-1)
        recon_loss, z_kl, agg_c_kl, c_entropy, u_kl, BC = ELBO_criterion(
            xhat, image, z_mean, z_logvar, c_logit, u_mean, u_logvar, model, num_classes, args
        )
        z_loss = z_kl
        c_loss = agg_c_kl
        c_entropy_loss = c_entropy
        u_loss = u_kl
        valid_BC = BC * BC_valid_mask
        prior_intersection_loss = tf.reduce_sum(valid_BC)
        loss = recon_loss + z_loss + c_loss + c_entropy_loss + u_loss + prior_intersection_loss
        
        loss_avg(loss)
        recon_loss_avg(recon_loss)
        z_loss_avg(z_loss)
        c_loss_avg(c_loss)
        c_entropy_loss_avg(c_entropy_loss)
        u_loss_avg(u_loss)
        prior_intersection_loss_avg(prior_intersection_loss)
        accuracy(tf.argmax(prob, axis=1, output_type=tf.int32), 
                 tf.argmax(label, axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split} Loss: {loss_avg.result():.4f}, Recon: {recon_loss_avg.result():.4f}, Z_Loss: {z_loss_avg.result():.4f}, C_Loss: {c_loss_avg.result():.4f}, C_Entropy: {c_entropy_loss_avg.result():.4f}, U_Loss: {u_loss_avg.result():.4f}, Prior_intersection: {prior_intersection_loss_avg.result():.4f}, Accuracy: {accuracy.result():.3%}')
    
    return loss_avg, recon_loss_avg, z_loss_avg, c_loss_avg, c_entropy_loss_avg, u_loss_avg, prior_intersection_loss_avg, accuracy
#%%
if __name__ == '__main__':
    main()
#%%