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
    tags=["svhn", "shotvae"],
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

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=600, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct-freq', '-rf', default=10, type=int,
                        metavar='N', help='reconstruct frequency (default: 10)')
    parser.add_argument('--labeled_examples', type=int, default=1000, 
                        help='number labeled examples (default: 1000')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')

    '''Deep VAE Model Parameters'''
    parser.add_argument('--depth', type=int, default=28, 
                        help='depth for WideResnet (default: 28)')
    parser.add_argument('--width', type=int, default=4, 
                        help='widen factor for WideResnet (default: 2)')
    parser.add_argument('--slope', type=float, default=0.1, 
                        help='slope parameter for LeakyReLU (default: 0.1)')
    parser.add_argument('--temperature', default=0.67, type=float,
                        help='centeralization parameter')
    parser.add_argument('-dr', '--drop-rate', default=0, type=float, 
                        help='drop rate for the network')
    parser.add_argument("--br", "--bce_reconstruction", default=True, type=bool,
                        help="Do BCE Reconstruction")
    parser.add_argument("-s", "--x-sigma", default=1, type=float,
                        help="The standard variance for reconstructed images, work as regularization")

    '''VAE parameters, notice we do not manually set the mutual information'''
    parser.add_argument('--ldc', "--latent-dim-continuous", default=128, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    parser.add_argument('--cmi', "--continuous-mutual-info", default=0, type=float,
                        help='The mutual information bounding between x and the continuous variable z')
    parser.add_argument('--dmi', "--discrete-mutual-info", default=2.3, type=float,
                        help='The mutual information bounding between x and the discrete variable z')

    '''VAE Loss Function Parameters'''
    parser.add_argument('--kbmc', '--kl-beta-max-continuous', default=1e-3, type=float, 
                        metavar='KL Beta', help='the epoch to linear adjust kl beta')
    parser.add_argument('--kbmd', '--kl-beta-max-discrete', default=1e-3, type=float, 
                        metavar='KL Beta', help='the epoch to linear adjust kl beta')
    parser.add_argument('--akb', '--adjust-kl-beta-epoch', default=200, type=int, 
                        metavar='KL Beta', help='the max epoch to adjust kl beta')
    parser.add_argument('--ewm', '--elbo-weight-max', default=1e-3, type=float, 
                        metavar='weight for elbo loss part')
    parser.add_argument('--aew', '--adjust-elbo-weight', default=400, type=int,
                        metavar="the epoch to adjust elbo weight to max")
    parser.add_argument('--wrd', default=1, type=float,
                        help="the max weight for the optimal transport estimation of discrete variable c")
    parser.add_argument('--wmf', '--weight-modify-factor', default=0.4, type=float,
                        help="weight  will get wrz at amf * epochs")
    parser.add_argument('--pwm', '--posterior-weight-max', default=1, type=float,
                        help="the max value for posterior weight")
    parser.add_argument('--apw', '--adjust-posterior-weight', default=200, type=float,
                        help="adjust posterior weight")

    '''Optimizer Parameters'''
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-b1', '--beta1', default=0.9, type=float, metavar='Beta1 In ADAM and SGD',
                        help='beta1 for adam as well as momentum for SGD')
    parser.add_argument('-ad', "--adjust-lr", default=[400, 500, 550], type=arg_as_list,
                        help="The milestone list for adjust learning rate")
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)

    '''Optimizer Transport Estimation Parameters'''
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help="the label smoothing epsilon for labeled data")
    parser.add_argument('--om', action='store_true', help="the optimal match for unlabeled data mixup")

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
def generate_and_save_images(model, image, num_classes, step, save_dir):
    z = model.get_latent(image, training=False)
    
    figure = plt.figure(figsize=(10, 2))
    plt.subplot(1, num_classes+1, 1)
    plt.imshow(image[0])
    plt.title('original')
    plt.axis('off')
    for i in range(num_classes):
        label = np.zeros((z.shape[0], num_classes))
        label[:, i] = 1
        xhat = model.decode_sample(z, label, training=False)
        plt.subplot(1, num_classes+1, i+2)
        plt.imshow(xhat[0])
        plt.title('{}'.format(i))
        plt.axis('off')
    # plt.savefig('{}/imgs/image_at_epoch_{}.png'.format(save_dir, step))
    plt.show()
    plt.close()
    return figure
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
    model = VAE(num_classes=num_classes, depth=args['depth'], width=args['width'], slope=args['slope'],
                latent_dim=args['ldc'], temperature=args['temperature'])
    model.build(input_shape=[(None, 32, 32, 3), (None, num_classes)])
    # model.summary()
    
    buffer_model = VAE(num_classes=num_classes, depth=args['depth'], width=args['width'], slope=args['slope'],
                    latent_dim=args['ldc'], temperature=args['temperature'])
    buffer_model.build(input_shape=[(None, 32, 32, 3), (None, num_classes)])
    buffer_model.set_weights(model.get_weights()) # weight initialization
    #%%
    '''
    <SGD + momentum + weight_decay>
    lambda: weight_decay parameter
    beta_1: momemtum
    lr: learning rate
    
    v(0) = 0
    for t in range(0, epochs):
        v(t+1) = beta_1 * v(t) + grad(t+1) + lambda * weight(t)
        weight(t+1) 
        = weight(t) - lr * v(t+1)
        = weight(t) - lr * (beta_1 * v(t) + grad(t+1) + lambda * weight(t))
        = (weight(t) - lr * grad(t+1)) - lr * (beta_1 * v(t) + lambda * weight(t))
        = (weight(t) - lr * (grad(t+1) + beta_1 * v(t)) - lr * lambda * weight(t))
    
    SGD : weight(t) - lr * grad(t+1)
    weight_decay (+ momentum) : - lr * (beta_1 * v(t) + lambda * weight(t))
    '''
    optimizer = K.optimizers.SGD(learning_rate=args['lr'],
                                momentum=args['beta1'])
    #%%
    for epoch in range(args['start_epoch'], args['epochs']):
        
        '''learning rate schedule'''
        if epoch == 0:
            '''warm-up'''
            optimizer.lr = args['lr'] * 0.2
        elif epoch < args['adjust_lr'][0]:
            optimizer.lr = args['lr']
        elif epoch < args['adjust_lr'][1]:
            optimizer.lr = args['lr'] * 0.1
        else:
            optimizer.lr = args['lr'] * 0.01
            
        # '''learning rate schedule'''
        # if epoch == 0:
        #     '''warm-up'''
        #     optimizer.lr = args['lr'] * 0.2
        # elif epoch < args['adjust_lr'][0]:
        #     optimizer.lr = args['lr']
        # elif epoch < args['adjust_lr'][1]:
        #     optimizer.lr = args['lr'] * 0.1
        # elif epoch < args['adjust_lr'][2]:
        #     optimizer.lr = args['lr'] * 0.01
        # else:
        #     optimizer.lr = args['lr'] * 0.001
        
        if epoch % args['reconstruct_freq'] == 0:
            labeled_loss, unlabeled_loss, kl_y_loss, accuracy, sample_recon = train(
                datasetL, datasetU, model, buffer_model, optimizer, epoch, args, num_classes, total_length, model_path
            )
            wandb.log({'(train) sample_recon': wandb.Image(sample_recon)})
        else:
            labeled_loss, unlabeled_loss, kl_y_loss, accuracy = train(
                datasetL, datasetU, model, buffer_model, optimizer, epoch, args, num_classes, total_length, model_path
            )
        val_z_kl_loss, val_y_kl_loss, val_recon_loss, val_elbo_loss, val_accuracy = validate(
            val_dataset, model, epoch, args, num_classes, split='Validation'
        )
        test_z_kl_loss, test_y_kl_loss, test_recon_loss, test_elbo_loss, test_accuracy = validate(
            test_dataset, model, epoch, args, num_classes, split='Test'
        )
        
        wandb.log({'(train) labeled_loss': labeled_loss.result().numpy()})
        wandb.log({'(train) unlabeled_loss': unlabeled_loss.result().numpy()})
        wandb.log({'(train) kl_y_loss': kl_y_loss.result().numpy()})
        wandb.log({'(train) accuracy': accuracy.result().numpy()})
        
        wandb.log({'(val) z_kl_loss': val_z_kl_loss.result().numpy()})
        wandb.log({'(val) y_kl_loss': val_y_kl_loss.result().numpy()})
        wandb.log({'(val) recon_loss': val_recon_loss.result().numpy()})
        wandb.log({'(val) elbo_loss': val_elbo_loss.result().numpy()})
        wandb.log({'(val) accuracy': val_accuracy.result().numpy()})
        
        wandb.log({'(test) z_kl_loss': test_z_kl_loss.result().numpy()})
        wandb.log({'(test) y_kl_loss': test_y_kl_loss.result().numpy()})
        wandb.log({'(test) recon_loss': test_recon_loss.result().numpy()})
        wandb.log({'(test) elbo_loss': test_elbo_loss.result().numpy()})
        wandb.log({'(test) accuracy': test_accuracy.result().numpy()})

        # Reset metrics every epoch
        labeled_loss.reset_states()
        unlabeled_loss.reset_states()
        kl_y_loss.reset_states()
        accuracy.reset_states()
        val_z_kl_loss.reset_states()
        val_y_kl_loss.reset_states()
        val_recon_loss.reset_states()
        val_elbo_loss.reset_states()
        val_accuracy.reset_states()
        test_z_kl_loss.reset_states()
        test_y_kl_loss.reset_states()
        test_recon_loss.reset_states()
        test_elbo_loss.reset_states()
        test_accuracy.reset_states()
        
        if epoch == 0:
            optimizer.lr = args['lr']
            
        if args['dataset'] == 'cifar10':
            if args['labeled_examples'] >= 2500:
                if epoch == args['adjust_lr'][0]:
                    args['ewm'] = args['ewm'] * 5
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
def train(datasetL, datasetU, model, buffer_model, optimizer, epoch, args, num_classes, total_length, model_path):
    labeled_loss_avg = tf.keras.metrics.Mean()
    unlabeled_loss_avg = tf.keras.metrics.Mean()
    kl_y_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    '''mutual information'''
    dmi = tf.convert_to_tensor(weight_schedule(epoch, args['akb'], args['dmi']), dtype=tf.float32)
    '''elbo part weight'''
    ew = weight_schedule(epoch, args['aew'], args['ewm'])
    '''mix-up parameters'''
    kl_beta_z = weight_schedule(epoch, args['akb'], args['kbmc'])
    kl_beta_y = weight_schedule(epoch, args['akb'], args['kbmd'])
    pwm = weight_schedule(epoch, args['apw'], args['pwm'])
    '''un-supervised classification weight'''
    ucw = weight_schedule(epoch, round(args['wmf'] * args['epochs']), args['wrd'])

    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'], drop_remainder=True)

    iteratorL = iter(shuffle_and_batch(datasetL))
    iteratorU = iter(shuffle_and_batch(datasetU))
    
    iteration = total_length // args['batch_size'] 
    
    progress_bar = tqdm.tqdm(range(iteration), unit='batch')
    for batch_num in progress_bar:
        
        '''augmentation'''
        try:
            imageL, labelL = next(iteratorL)
            imageL = augment(imageL)
        except:
            iteratorL = iter(shuffle_and_batch(datasetL))
            imageL, labelL = next(iteratorL)
            imageL = augment(imageL)
        imageU, _ = next(iteratorU)
        imageU = augment(imageU)
        
        '''mix-up weight'''
        mix_weight = [tf.constant(np.random.beta(args['epsilon'], args['epsilon'])), # labeled
                      tf.constant(np.random.beta(2.0, 2.0))] # unlabeled
        
        '''labeled'''
        with tf.GradientTape() as tape:
            meanL, log_sigmaL, log_probL, _, _, xhatL = model([imageL, labelL])
            recon_lossL, kl_zL, kl_yL = ELBO_criterion(args, num_classes, imageL, xhatL, meanL, log_sigmaL, log_probL)
            prior_klL = (kl_beta_z * kl_zL) + (kl_beta_y * tf.math.abs(kl_yL - dmi))
            elbo_lossL = recon_lossL + prior_klL
            
            '''mix-up''' 
            with tape.stop_recording():
                image_mixL, label_shuffleL, mean_mixL, sigma_mixL = label_smoothing(imageL, labelL, meanL, log_sigmaL, mix_weight[0])
            smoothed_meanL, smoothed_log_sigmaL, smoothed_log_probL, _, _, _ = model([image_mixL, label_shuffleL])
            
            posterior_loss_zL = tf.reduce_mean(tf.math.square(smoothed_meanL - mean_mixL))
            posterior_loss_zL += tf.reduce_mean(tf.math.square(tf.math.exp(smoothed_log_sigmaL) - sigma_mixL))
            posterior_loss_yL = - tf.reduce_mean(mix_weight[0] * tf.reduce_sum(label_shuffleL * smoothed_log_probL, axis=-1))
            posterior_loss_yL += - tf.reduce_mean((1. - mix_weight[0]) * tf.reduce_sum(labelL * smoothed_log_probL, axis=-1))
            
            elbo_lossL += kl_beta_z * pwm * posterior_loss_zL
            loss_supervised = (ew * elbo_lossL) + posterior_loss_yL

        grads = tape.gradient(loss_supervised, model.trainable_variables) 
        '''SGD + momentum''' 
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        '''decoupled weight decay'''
        weight_decay_decoupled(model, buffer_model, decay_rate=args['wd'] * optimizer.lr)
        
        '''unlabeled'''
        with tf.GradientTape() as tape:
            meanU, log_sigmaU, log_probU, _, _, xhatU = model([imageU, None])
            recon_lossU, kl_zU, kl_yU = ELBO_criterion(args, num_classes, imageU, xhatU, meanU, log_sigmaU, log_probU)
            prior_klU = (kl_beta_z * kl_zU) + (kl_beta_y * tf.math.abs(kl_yU - dmi))
            elbo_lossU = recon_lossU + prior_klU
            
            '''mix-up'''
            with tape.stop_recording():
                image_mixU, mean_mixU, sigma_mixU, pseudo_labelU = optimal_match_mix(imageU, meanU, log_sigmaU, log_probU, mix_weight[1], args['om'])
            smoothed_meanU, smoothed_log_sigmaU, smoothed_log_probU, _, _, _ = model([image_mixU, _])
            
            posterior_loss_zU = tf.reduce_mean(tf.math.square(smoothed_meanU - mean_mixU))
            posterior_loss_zU += tf.reduce_mean(tf.math.square(tf.math.exp(smoothed_log_sigmaU) - sigma_mixU))
            posterior_loss_yU = - tf.reduce_mean(tf.reduce_sum(pseudo_labelU * smoothed_log_probU, axis=-1))
            
            elbo_lossU += kl_beta_z * pwm * posterior_loss_zU
            loss_unsupervised = (ew * elbo_lossU) + (ucw * posterior_loss_yU)

        grads = tape.gradient(loss_unsupervised, model.trainable_variables) 
        '''SGD + momentum''' 
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        '''decoupled weight decay'''
        weight_decay_decoupled(model, buffer_model, decay_rate=args['wd'] * optimizer.lr)
        
        labeled_loss_avg(loss_supervised)
        unlabeled_loss_avg(loss_unsupervised)
        kl_y_loss_avg(kl_yU)
        _, _, log_probL, _, _, _ = model([imageL, labelL], training=False)
        accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), log_probL)

        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'labeled Loss': f'{labeled_loss_avg.result():.4f}',
            'unlabeled Loss': f'{unlabeled_loss_avg.result():.4f}',
            'KL': f'{kl_y_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}'
        })
    
    if epoch % args['reconstruct_freq'] == 0:
        sample_recon = generate_and_save_images(model, imageU[0][tf.newaxis, ...], num_classes, epoch, model_path)
        return labeled_loss_avg, unlabeled_loss_avg, kl_y_loss_avg, accuracy, sample_recon
    else:
        return labeled_loss_avg, unlabeled_loss_avg, kl_y_loss_avg, accuracy
#%%
def validate(dataset, model, epoch, args, num_classes, split):
    z_kl_loss_avg = tf.keras.metrics.Mean()
    y_kl_loss_avg = tf.keras.metrics.Mean()
    recon_loss_avg = tf.keras.metrics.Mean()   
    elbo_loss_avg = tf.keras.metrics.Mean()   
    accuracy = tf.keras.metrics.Accuracy()

    dataset = dataset.batch(args['batch_size'])
    for image, label in dataset:
        mean, log_sigma, log_prob, _, _, xhat = model([image, None], training=False)
        recon_loss, kl_z, kl_y = ELBO_criterion(args, num_classes, image, xhat, mean, log_sigma, log_prob)
        z_kl_loss_avg(kl_z)
        y_kl_loss_avg(kl_y)
        recon_loss_avg(recon_loss)
        elbo_loss_avg(recon_loss + 0.01 * (kl_z + kl_y))
        accuracy(tf.argmax(log_prob, axis=1, output_type=tf.int32), 
                 tf.argmax(label, axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split} ELBO Loss: {elbo_loss_avg.result():.4f}, KL(z): {z_kl_loss_avg.result():.4f}, KL(y): {y_kl_loss_avg.result():.4f} {split} Accuracy: {accuracy.result():.3%}')
    
    return z_kl_loss_avg, y_kl_loss_avg, recon_loss_avg, elbo_loss_avg, accuracy
#%%
def weight_schedule(epoch, epochs, weight_max):
    return weight_max * tf.math.exp(-5. * (1. - min(1., epoch/epochs)) ** 2)
#%%
if __name__ == '__main__':
    main()
#%%