#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
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
class Decoder(K.models.Model):
    def __init__(self, 
                 output_channel, 
                 activation, 
                 name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.num_feature = 64
        self.units = self._build_unit(output_channel, activation)
    
    def _build_unit(self, output_channel, activation):
        dims = [self.num_feature * d for d in [16, 8, 4, 2]]
        units = []
        for i in range(len(dims)):
            units.append(layers.Conv2DTranspose(filters = dims[i], kernel_size = 5, strides = 2, 
                                                padding = 'same', use_bias=False))
            units.append(layers.BatchNormalization())
            units.append(layers.ReLU())
        units.append(layers.Conv2DTranspose(filters = output_channel, kernel_size = 4, strides = 2, 
                                            activation=activation,
                                            padding = 'same', use_bias=False))
        return K.models.Sequential(units)
    
    @tf.function
    def call(self, x, training=True):
        h = x[:, tf.newaxis, tf.newaxis, :]
        h = self.units(h, training=training)
        return h
#%%
class VAE(K.models.Model):
    def __init__(self, 
                 num_classes=10,
                 depth=28,
                 width=2,
                 slope=0.1,
                 latent_dim=128, 
                 output_channel=3, 
                 activation='sigmoid',
                 temperature=1.,
                 input_shape=(None, 32, 32, 3), 
                 name='VAE', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        
        self.FeatureExtractor = WideResNet(depth, width, slope, input_shape)
        self.mean_layer = layers.Dense(latent_dim) 
        self.logsigma_layer = layers.Dense(latent_dim) 
        self.prob_layer = layers.Dense(num_classes) 
        self.Decoder = Decoder(output_channel, activation)
        
        self.temperature = temperature
        self.latent_dim = latent_dim
        
    def get_latent(self, x, training=False):
        h = self.FeatureExtractor(x, training=training)
        mean = self.mean_layer(h)
        log_sigma = self.logsigma_layer(h)
        noise = tf.random.normal((tf.shape(x)[0], self.latent_dim))
        z = mean + tf.math.exp(log_sigma) * noise 
        return z
    
    def decode_sample(self, z, y, training=False):
        return self.Decoder(tf.concat([z, y], axis=-1), training=training) 
        
    def sample_gumbel(self, shape): 
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + 1e-8) + 1e-8)

    def gumbel_max_sample(self, log_prob): 
        y = log_prob + self.sample_gumbel(tf.shape(log_prob))
        y = tf.nn.softmax(y / self.temperature)
        return y
    
    @tf.function
    def call(self, inputs, training=True):
        x, label = inputs
        h = self.FeatureExtractor(x, training=training)
        mean = self.mean_layer(h)
        log_sigma = self.logsigma_layer(h)
        log_prob = tf.nn.log_softmax(self.prob_layer(h))
        y = self.gumbel_max_sample(log_prob)
        
        noise = tf.random.normal((tf.shape(x)[0], self.latent_dim))
        z = mean + tf.math.exp(log_sigma) * noise 
        
        '''for labeled'''
        if tf.is_tensor(label):
            y = label
        
        xhat = self.Decoder(tf.concat([z, y], axis=-1), training=training) 
        return mean, log_sigma, log_prob, z, y, xhat
#%%