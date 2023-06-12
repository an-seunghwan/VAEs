#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class Encoder(K.models.Model):
    def __init__(self, latent_dim, name="Encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.feature_num = 32
        self.nChannels = [self.feature_num * d for d in [1, 2, 4, 8]]
        
        self.net = K.Sequential(
            [
                layers.Conv2D(filters=self.nChannels[0], kernel_size=5, strides=2, padding='same'), # 16x16
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Conv2D(filters=self.nChannels[1], kernel_size=4, strides=2, padding='same'), # 8x8
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Conv2D(filters=self.nChannels[2], kernel_size=4, strides=2, padding='same'), # 4x4
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Conv2D(filters=self.nChannels[3], kernel_size=4, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Flatten(),
                layers.Dense(1024),
                layers.BatchNormalization(),
            ]
        )
        self.mean_layer = layers.Dense(latent_dim, activation='linear')
        self.logvar_layer = layers.Dense(latent_dim, activation='softplus')
        
    @tf.function
    def call(self, inputs, training=True):
        x, y = inputs
        h = self.net(x, training=training)
        mean = self.mean_layer(tf.concat([h, y], axis=-1))
        logvar = self.logvar_layer(tf.concat([h, y], axis=-1))
        return mean, logvar
#%%
class Classifier(K.models.Model):
    def __init__(self, 
                 num_classes,
                 dropratio=0.1,
                 name="Classifier", **kwargs):
        super(Classifier, self).__init__(name=name, **kwargs)
        self.units = K.Sequential(
            [
                layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                layers.SpatialDropout2D(rate=dropratio),
                
                layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                layers.SpatialDropout2D(rate=dropratio),
                
                layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.GlobalAveragePooling2D(),
                
                layers.Dense(num_classes, activation='softmax')
            ]
        )
    
    @tf.function
    def call(self, x, training=True):
        h = self.units(x, training=training)
        return h
#%%
class Decoder(K.models.Model):
    def __init__(self, channel, activation, name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.feature_num = 32
        self.nChannels = [self.feature_num * d for d in [8, 4, 2, 1]]
        
        self.net = K.Sequential(
            [
                layers.Dense(4*4*512),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Reshape((4, 4, 512)),
                
                layers.Conv2DTranspose(filters=self.nChannels[0], kernel_size=5, strides=2, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Conv2DTranspose(filters=self.nChannels[1], kernel_size=5, strides=2, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Conv2DTranspose(filters=self.nChannels[2], kernel_size=5, strides=2, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Conv2DTranspose(filters=self.nChannels[3], kernel_size=5, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Conv2D(filters=channel, kernel_size=4, strides=1, padding='same', activation=activation)
            ]
        )
    
    @tf.function
    def call(self, x, training=True):
        h = self.net(x, training=training)
        return h
#%%
class DGM(K.models.Model):
    def __init__(self, 
                 num_classes=10,
                 latent_dim=128, 
                 output_channel=3, 
                 dropratio=0.1,
                 activation='sigmoid',
                 input_dim=(None, 32, 32, 3), 
                 name='DGM', **kwargs):
        super(DGM, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        self.encoder = Encoder(latent_dim)
        self.classifier = Classifier(num_classes, dropratio)
        self.decoder = Decoder(output_channel, activation)
    
    def encode(self, inputs, training=True):
        x, y = inputs
        mean, logvar = self.encoder([x, y], training=training)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.latent_dim))
        z = mean + tf.math.exp(logvar / 2.) * epsilon 
        return mean, logvar, z
    
    def classify(self, x, training=True):
        prob = self.classifier(x, training=training)
        return prob
    
    def decode(self, z, y, training=True):
        xhat = self.decoder(tf.concat([z, y], axis=-1), training=training) 
        return xhat
        
    @tf.function
    def call(self, inputs, training=True):
        x, y = inputs
        mean, logvar = self.encoder([x, y], training=training)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.latent_dim))
        z = mean + tf.math.exp(logvar / 2.) * epsilon 
        xhat = self.decoder(tf.concat([z, y], axis=-1), training=training) 
        return mean, logvar, z, xhat
#%%
# x = tf.random.normal((10, 32, 32, 3))
# y = tf.random.normal((10, 10))
# dgm = DGM()
# hs = dgm([x, y])
# for h in hs:
#     print(h.shape)
#%%