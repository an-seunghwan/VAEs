# VAE (or GAN) Methods

This repository is unofficial implementation of following papers with Tensorflow 2.0 or pytorch. The corresponding folder name is written in parenthesis.

- Disentangling:
  - [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) (`betavae`)

- Representation:
  - [Neural discrete representation learning](https://proceedings.neurips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf) (`vqvae`)
  - [Weakly Supervised Disentangled Generative Causal Representation Learning](https://arxiv.org/abs/2010.02637) (`dear`)

- Semi-Supervised:
  - [Semi-supervised Learning with Deep Generative Models](https://proceedings.neurips.cc/paper/2014/hash/d523773c6b194f37b938d340d5d02232-Abstract.html) (`dgm`)
  - [Auxiliary deep generative models](http://proceedings.mlr.press/v48/maaloe16.html) (`adgm`)
  - [Ladder variational autoencoders](https://proceedings.neurips.cc/paper/2016/file/6ae07dcb33ec3b7c814df797cbda0f87-Paper.pdf) (`ladder`)
  - [Semi-supervised disentanglement of class-related and class-independent factors in vae](https://arxiv.org/pdf/2102.00892.pdf) (`partedvae`)
  - [SHOT-VAE: semi-supervised deep generative models with label-aware ELBO approximations](https://www.aaai.org/AAAI21Papers/AAAI-260.FengHZ.pdf) (`shotvae`)

## Package Dependencies

```setup
tensorflow==2.4.0
pytorch==1.12.0
```
Additional package requirements for this repository are described in `requirements.txt`.

## How to Training & Evaluation  

`labeled_examples` is the number of labeled datsets for running and we provide configuration `.yaml` files for 100 labeled datsets of MNIST and 4000 labeled datasets of CIFAR-10. And we add required tests and evaluations at the end of code.

1. MNIST dataset running

```
python mnist/main.py --config_path "configs/mnist_{labeled_examples}.yaml"
```   

2. CIFAR-10 dataset running

```
python main.py --config_path "configs/cifar10_{labeled_examples}.yaml"
```   

## Results (CIFAR-10)

The number in parenthesis next to the name of model is the number of parameters in classifier. Inception score of classification model is not computed.

|       Model      | Classification error | Inception Score |
|:----------------:|:--------------------:|:---------------:|
| M2(4.5M)         |               27.69% |     1.85 (0.05) |
| Parted-VAE(5.8M) |               31.85% |      1.58(0.04) |
| SHOT-VAE(5.8M)   |                5.91% |     3.46 (0.18) |

## Reference codes

- https://github.com/wohlert/semi-supervised-pytorch
- https://github.com/sinahmr/parted-vae
- https://github.com/FengHZ/AAAI2021-260
- https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
