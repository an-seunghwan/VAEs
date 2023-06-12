#%%
import argparse
import os

os.chdir(r'D:\semi\shotvae') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/shotvae') # main directory (repository)

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tqdm
import yaml
import io
import matplotlib.pyplot as plt
from PIL import Image

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

from preprocess import fetch_dataset
from model import VAE
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

    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used for training (e.g. cifar10, cifar100, svhn, svhn+extra)')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=600, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct-freq', '-rf', default=50, type=int,
                        metavar='N', help='reconstruct frequency (default: 50)')
    parser.add_argument('--labeled_examples', type=int, default=4000, 
                        help='number labeled examples (default: 4000')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')

    '''Deep VAE Model Parameters'''
    parser.add_argument('--depth', type=int, default=28, 
                        help='depth for WideResnet (default: 28)')
    parser.add_argument('--width', type=int, default=2, 
                        help='widen factor for WideResnet (default: 2)')
    parser.add_argument('--slope', type=float, default=0.1, 
                        help='slope parameter for LeakyReLU (default: 0.1)')
    parser.add_argument('--temperature', default=0.67, type=float,
                        help='centeralization parameter')
    parser.add_argument('-dr', '--drop-rate', default=0, type=float, 
                        help='drop rate for the network')
    parser.add_argument("--br", "--bce-reconstruction", action='store_true', 
                        help='Do BCE Reconstruction')
    parser.add_argument("-s", "--x-sigma", default=1, type=float,
                        help="The standard variance for reconstructed images, work as regularization")

    '''VAE parameters, notice we do not manually set the mutual information'''
    parser.add_argument('--ldc', "--latent-dim-continuous", default=128, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    parser.add_argument('--cmi', "--continuous-mutual-info", default=0, type=float,
                        help='The mutual information bounding between x and the continuous variable z')
    parser.add_argument('--dmi', "--discrete-mutual-info", default=0, type=float,
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

    return parser
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
args = vars(get_args().parse_args(args=['--config_path', 'configs/cifar10_4000.yaml']))

dir_path = os.path.dirname(os.path.realpath(__file__))
if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
    args = load_config(args)

log_path = f'logs/{args["dataset"]}_{args["labeled_examples"]}'

datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)

# model_path = r'D:\semi\shotvae\logs\cifar10_4000\7.change_dataset\seed1\20220120-191853'
model_path = r'D:\semi\shotvae\logs\cifar10_4000\20220414-110056'
model_name = [x for x in os.listdir(model_path) if x.endswith('.h5')][0]
model = VAE(num_classes=num_classes, depth=args['depth'], width=args['width'], slope=args['slope'],
            latent_dim=args['ldc'], temperature=args['temperature'])
model.build(input_shape=[(None, 32, 32, 3), (None, num_classes)])
model.load_weights(model_path + '/' + model_name)
model.summary()
#%%
classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
classdict = {i:x for i,x in enumerate(classnames)}
#%%
# shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'], drop_remainder=True)
# iteratorL = iter(shuffle_and_batch(datasetL))
# image, label = next(iteratorL)
#%%
data_dir = r'D:\cifar10_{}'.format(5000)
idx = np.arange(100)
x = np.array([np.load(data_dir + '/x_{}.npy'.format(i)) for i in idx])
y = np.array([np.load(data_dir + '/y_{}.npy'.format(i)) for i in idx])
x = tf.cast(x, tf.float32) / 255.

mean, log_sigma, log_prob, z, y, xhat = model([x, y], training=False)
#%%
for idx in tqdm.tqdm(range(100)):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, num_classes+1, 1)
    plt.imshow(x[idx])
    plt.title('original')
    plt.axis('off')
    
    for i in range(num_classes):
        label = np.zeros((1, num_classes))
        label[:, i] = 1
        label = tf.cast(label, tf.float32)
        xhat = model.decode_sample(z.numpy()[[idx]], label, training=False)

        plt.subplot(1, num_classes+1, i+2)
        plt.imshow(xhat[0])
        plt.title('{}'.format(classdict.get(i)), fontsize=15)
        plt.axis('off')
    plt.savefig('{}/img{}.png'.format(model_path, idx),
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    # plt.show()
    plt.close()
#%%
'''style transfer'''
style = []
# for idx in [1, 17, 21, 31, 32, 48, 58, 68, 81, 84]:
for idx in [11, 3, 15, 23, 34, 81]:
    style.append(Image.open('{}/img{}.png'.format(model_path, idx)))
    
fig, axes = plt.subplots(6, 1, figsize=(10, 6))
for i in range(len(style)):
    axes.flatten()[i].imshow(style[i])
    axes.flatten()[i].axis('off')
plt.tight_layout()
plt.savefig('{}/style_transfer.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''interpolation'''
pairs = [[11, 25], [11, 15], [11, 53]]
fig, axes = plt.subplots(3, 10, figsize=(10, 3))
for k in range(len(pairs)):
    z_interpolation = np.squeeze(np.linspace(z.numpy()[[pairs[k][0]], :], 
                                            z.numpy()[[pairs[k][1]], :], 8))

    label = np.zeros((z_interpolation.shape[0], num_classes))
    label[:, 1] = 1 # automobile
    xhat_ = model.decode_sample(z_interpolation, label, training=False)

    axes[k][0].imshow(x[pairs[k][0]])
    axes[k][0].axis('off')
    for i in range(len(z_interpolation)):
        axes[k][i+1].imshow(xhat_[i])
        axes[k][i+1].axis('off')
    axes[k][-1].imshow(x[pairs[k][1]])
    axes[k][-1].axis('off')

plt.tight_layout()
plt.savefig('{}/style_interpolation_nonsmooth.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''manipulation'''
idx = [11, 3, 15, 23, 34, 81]
plt.figure(figsize=(10, 6))
for j in range(len(idx)):
    z_ = z.numpy()[[idx[j]]]
    p = np.linspace(1, 0, 11)
    num1 = 1 # automobile
    num2 = 7 # horse

    plt.subplot(len(idx), len(p)+1, (len(p)+1) * j + 1)
    plt.imshow(x[idx[j]])
    plt.title('original')
    plt.axis('off')
    for i in range(len(p)):
        attr = np.zeros((1, num_classes))
        attr[:, num1] = p[i]
        attr[:, num2] = 1 - p[i]
        attr = tf.cast(attr, tf.float32)
        xhat = model.decode_sample(z_, attr, training=False)

        plt.subplot(len(idx), len(p)+1, (len(p)+1) * j + i + 2)
        plt.imshow(xhat[0])
        plt.title('{}:{:.1f}'.format(classdict.get(num1)[:4], p[i]), fontsize=12)
        plt.axis('off')
plt.tight_layout()
plt.savefig('{}/manipulation.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''style latent random sampling of z'''
tf.random.set_seed(10)
z_sampling = tf.random.normal(shape=(100, args['ldc']))
pi = np.zeros((len(z_sampling), num_classes))
pi[:, 1] = 1 # automobile
pi = tf.cast(pi, tf.float32)
xhat = model.decode_sample(z, pi, training=False)

fig, axes = plt.subplots(10, 10, figsize=(15, 15))
for i in range(100):
    axes.flatten()[i].imshow(xhat[i])
    axes.flatten()[i].axis('off')
plt.tight_layout()
plt.savefig('{}/z_sampling_{}.png'.format(model_path, classdict.get(1)),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''inception score'''
def calculate_inception_score(images, n_split=50, eps=1E-16):
    inception = K.applications.InceptionV3(include_top=True)
    scores = list()
    n_part = int(np.floor(images.shape[0] / n_split))
    for i in tqdm.tqdm(range(n_split)):
        ix_start, ix_end = i * n_part, (i+1) * n_part
        subset = images[ix_start:ix_end]
        subset = subset.astype('float32')
        # scale images to the required size
        subset = tf.image.resize(subset, (299, 299), 'nearest')
        # pre-process images, scale to [-1,1]
        subset = 2. * subset - 1.
        p_yx = inception.predict(subset)
        p_y = tf.expand_dims(p_yx.mean(axis=0), 0)
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = np.mean(sum_kl_d)
        is_score = np.exp(avg_kl_d)
        scores.append(is_score)
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std
#%%
# 10,000 generated images from sampled latent variables
np.random.seed(1)
generated_images = []
for i in tqdm.tqdm(range(num_classes)):
    for _ in range(10):
        latents = np.random.normal(size=(100, args['ldc']))
        y_ = np.zeros((100, num_classes))
        y_[:, i] = 1.
        images = model.decode_sample(latents, y_, training=False)
        generated_images.extend(images)
generated_images = np.array(generated_images)
np.random.shuffle(generated_images)
#%%
# calculate inception score
is_avg, is_std = calculate_inception_score(generated_images)
print('inception score | mean: {:.2f}, std: {:.2f}'.format(is_avg, is_std))
#%%
'''test dataset classification error'''
autotune = tf.data.AUTOTUNE
batch = lambda dataset: dataset.batch(batch_size=args['batch_size'], drop_remainder=False).prefetch(autotune)
total_length = sum(1 for _ in test_dataset)
iteration = total_length // args['batch_size'] 

error_count = 0
for x_test_batch, y_test_batch in batch(test_dataset):
    _, _, log_prob, _, _, _ = model([x_test_batch, y_test_batch], training=False)
    error_count += np.sum(tf.argmax(log_prob, axis=-1).numpy() - tf.argmax(y_test_batch, axis=-1).numpy() != 0)
print('TEST classification error: {:.2f}%'.format(error_count / total_length * 100))
#%%
with open('{}/result.txt'.format(model_path), "w") as file:
    file.write('TEST classification error: {:.2f}%\n\n'.format(error_count / total_length * 100))
    file.write('inception score | mean: {:.2f}, std: {:.2f}\n\n'.format(is_avg, is_std))
#%%
'''interpolation (comparison)'''
data_dir = r'D:\cifar10_{}'.format(5000)
idx = np.arange(100)
x = np.array([np.load(data_dir + '/x_{}.npy'.format(i)) for i in idx])
y = np.array([np.load(data_dir + '/y_{}.npy'.format(i)) for i in idx])
x = tf.cast(x, tf.float32) / 255.

_, _, _, z, _, xhat = model([x, y], training=False)
#%%
fig, axes = plt.subplots(2, 10, figsize=(25, 5))
for idx, (class_idx, i, j) in enumerate([[1, 0, 5], [7, 0, 2]]):
    interpolation_idx = np.where(np.argmax(y, axis=-1) == class_idx)[0]

    inter = np.linspace(z[interpolation_idx[i]], z[interpolation_idx[j]], 8)
    inter_recon = model.decode_sample(inter, np.tile(y[[interpolation_idx[i]], :], (8, 1)), training=False)

    axes.flatten()[idx*10 + 0].imshow(x[interpolation_idx[i]])
    axes.flatten()[idx*10 + 0].axis('off')
    for i in range(8):
        axes.flatten()[idx*10 + i+1].imshow(inter_recon[i].numpy())
        axes.flatten()[idx*10 + i+1].axis('off')
    axes.flatten()[idx*10 + 9].imshow(x[interpolation_idx[j]])
    axes.flatten()[idx*10 + 9].axis('off')
plt.savefig('{}/shotvae_interpolation1.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%