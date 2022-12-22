#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class ResidualUnit(K.layers.Layer):
    def __init__(self, 
                 filter_in, 
                 filter_out,
                 strides, 
                 slope=0.1,
                 **kwargs):
        super(ResidualUnit, self).__init__(**kwargs)
        
        self.norm1 = layers.BatchNormalization()
        self.relu1 = layers.LeakyReLU(alpha=slope)
        self.conv1 = layers.Conv2D(filters=filter_out, kernel_size=3, strides=strides, 
                                    padding='same', use_bias=False)
        self.norm2 = layers.BatchNormalization()
        self.relu2 = layers.LeakyReLU(alpha=slope)
        self.conv2 = layers.Conv2D(filters=filter_out, kernel_size=3, strides=1, 
                                    padding='same', use_bias=False)
        
        self.downsample = (filter_in != filter_out)
        if self.downsample:
            self.shortcut = layers.Conv2D(filters=filter_out, kernel_size=1, strides=strides, 
                                        padding='same', use_bias=False)

    @tf.function
    def call(self, x, training=True):
        if self.downsample:
            x = self.relu1(self.norm1(x, training=training))
            h = self.relu2(self.norm2(self.conv1(x), training=training))
        else:
            h = self.relu1(self.norm1(x, training=training))
            h = self.relu2(self.norm2(self.conv1(h), training=training))
        h = self.conv2(h)
        if self.downsample:
            h = h + self.shortcut(x)
        else:
            h = h + x
        return h
#%%
class ResidualBlock(K.layers.Layer):
    def __init__(self,
                 n_units,
                 filter_in,
                 filter_out,
                 unit,
                 strides, 
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units = self._build_unit(n_units, unit, filter_in, filter_out, strides)
    
    def _build_unit(self, n_units, unit, filter_in, filter_out, strides):
        units = []
        for i in range(n_units):
            units.append(unit(filter_in if i == 0 else filter_out, filter_out, strides if i == 0 else 1))
        return K.models.Sequential(units)
    
    @tf.function
    def call(self, x, training=True):
        x = self.units(x, training=training)
        return x
#%%
class WideResNet(K.models.Model):
    def __init__(self, 
                 depth=28,
                 width=2,
                 slope=0.1,
                 input_shape=(None, 32, 32, 3),
                 name="WideResNet", 
                 **kwargs):
        super(WideResNet, self).__init__(input_shape, name=name, **kwargs)
        
        assert (depth - 4) % 6 == 0
        self.n_units = (depth - 4) // 6
        self.nChannels = [16, 16*width, 32*width, 64*width]
        self.slope = slope
        
        '''small_input = True'''
        self.conv = layers.Conv2D(filters=self.nChannels[0], kernel_size=3, strides=1, 
                                    padding='same', use_bias=False)
        
        self.block1 = ResidualBlock(self.n_units, self.nChannels[0], self.nChannels[1], ResidualUnit, 1)
        self.block2 = ResidualBlock(self.n_units, self.nChannels[1], self.nChannels[2], ResidualUnit, 2)
        self.block3 = ResidualBlock(self.n_units, self.nChannels[2], self.nChannels[3], ResidualUnit, 2)
        
        self.norm = layers.BatchNormalization()
        self.relu = layers.LeakyReLU(alpha=slope)
        self.pooling = layers.GlobalAveragePooling2D()
    
    @tf.function
    def call(self, x, training=True):
        h = self.conv(x)
        h = self.block1(h, training=training)
        h = self.block2(h, training=training)
        h = self.block3(h, training=training)
        h = self.relu(self.norm(h, training=training))
        h = self.pooling(h)
        return h
#%%
# e = WideResNet(256)
# e.build((10, 32, 32, 3))
# e.summary()
# e(tf.random.normal((10, 32, 32, 3))).shape
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
# e = Decoder(1, 'sigmoid', 256)
# e.build((10, 12))
# e.net.summary()
#%%
class VAE(K.models.Model):
    def __init__(self, 
                num_classes=10,
                latent_dim=64, 
                u_dim=64,
                output_channel=3, 
                depth=28,
                width=2,
                slope=0.1,
                temperature=0.67,
                sigmoid_coef=1,
                activation='sigmoid',
                input_dim=(None, 32, 32, 3), 
                name='VAE', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.hidden_dim = width * 64
        self.u_dim = u_dim
        self.input_dim = input_dim
        self.temperature = temperature
        self.sigmoid_coef = sigmoid_coef
        
        self.feature_extractor = WideResNet(
            depth=depth,
            width=width,
            slope=slope,
            input_shape=input_dim
        )
        
        self.z_mean_layer = layers.Dense(latent_dim, activation='linear')
        self.z_logvar_layer = layers.Dense(latent_dim, activation='linear')
        
        self.h_to_c_logit = layers.Dense(num_classes, activation='linear') # h to c logit
        self.c_to_a_logit = layers.Dense(self.hidden_dim, activation='linear') # c to a logit
        
        self.u_mean_layer = layers.Dense(u_dim, activation='linear')
        self.u_logvar_layer = layers.Dense(u_dim, activation='linear')
        
        self.decoder = Decoder(output_channel, activation)
        
        self.u_prior_means = self.add_weight(shape=(num_classes, u_dim),
                                            initializer='random_normal',
                                            trainable=True,
                                            name='u_prior_means')
        self.u_prior_logvars_before_tanh = self.add_weight(shape=(num_classes, u_dim),
                                                            initializer='random_normal',
                                                            trainable=True,
                                                            name='u_prior_logvars_before_tanh')
    
    @property
    def u_prior_logvars(self):
        return 2. * tf.nn.tanh(self.u_prior_logvars_before_tanh) - 1.
    
    def sample_gumbel(self, shape): 
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + 1e-8) + 1e-8)

    def gumbel_softmax_sample(self, log_prob, training=True): 
        y = log_prob + self.sample_gumbel(tf.shape(log_prob))
        if training:
            y = tf.nn.softmax(y / self.temperature)
        else:
            y = tf.cast(tf.equal(y, tf.math.reduce_max(y, axis=1, keepdims=True)), y.dtype)
        return y
    
    def _sigmoid(self, x, training=True):
        if not training or self.sigmoid_coef > 8.:
            return tf.nn.sigmoid(8. * x)
        if self.sigmoid_coef < 8:
            self.sigmoid_coef += 2e-4
        return tf.nn.sigmoid(self.sigmoid_coef * x)
    
    def classify(self, x, training=True):
        hidden = self.feature_extractor(x, training=training)
        c_logit = self.h_to_c_logit(hidden)
        return tf.nn.softmax(c_logit, axis=-1)
    
    def encode(self, x, training=True):
        hidden = self.feature_extractor(x, training=training)
        
        # class independent
        z_mean = self.z_mean_layer(hidden)
        z_logvar = self.z_logvar_layer(hidden)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.latent_dim))
        z = z_mean + tf.math.exp(z_logvar / 2.) * epsilon 
        
        # class related
        c_logit = self.h_to_c_logit(hidden)
        onehot_c = self.gumbel_softmax_sample(c_logit, training=training)
        a_logit = self.c_to_a_logit(onehot_c)
        a = self._sigmoid(a_logit)
        hidden_a = hidden * a
        u_mean = self.u_mean_layer(hidden_a)
        u_logvar = self.u_logvar_layer(hidden_a)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.u_dim))
        u = u_mean + tf.math.exp(u_logvar / 2.) * epsilon 
        return z_mean, z_logvar, z, u_mean, u_logvar, u
    
    @tf.function
    def call(self, x, training=True):
        hidden = self.feature_extractor(x, training=training)
        
        # class independent
        z_mean = self.z_mean_layer(hidden)
        z_logvar = self.z_logvar_layer(hidden)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.latent_dim))
        z = z_mean + tf.math.exp(z_logvar / 2.) * epsilon 
        
        # class related
        c_logit = self.h_to_c_logit(hidden)
        onehot_c = self.gumbel_softmax_sample(c_logit, training=training)
        a_logit = self.c_to_a_logit(onehot_c)
        a = self._sigmoid(a_logit)
        hidden_a = hidden * a
        u_mean = self.u_mean_layer(hidden_a)
        u_logvar = self.u_logvar_layer(hidden_a)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.u_dim))
        u = u_mean + tf.math.exp(u_logvar / 2.) * epsilon 
        
        xhat = self.decoder(tf.concat([z, u], axis=-1), training=training) 
        return z_mean, z_logvar, z, c_logit, u_mean, u_logvar, u, xhat
#%%
# model = VAE()
# model.build((10, 32, 32, 3))
# #%%
# x = tf.random.normal((16, 32, 32, 3))
# outputs = model(x)
# [t.shape for t in outputs]
# model.u_prior_means.shape
# model.u_prior_logvars.shape
# model.feature_extractor.summary()
# model.feature_extractor.trainable_variables + model.h_to_c_logit.trainable_variables
#%%