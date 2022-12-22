#%%
import numpy as np
import tensorflow as tf
#%%
def ELBO_criterion(xhat, x, y, z, mean, logvar, num_classes, args):
    if args['bce_reconstruction']:
        error = - tf.reduce_sum(x * tf.math.log(tf.clip_by_value(xhat, 1e-10, 1.)) + 
                                (1. - x) * tf.math.log(1. - tf.clip_by_value(xhat, 1e-10, 1.)), axis=[1, 2, 3])
    else:
        error = tf.reduce_sum(tf.math.square(x - xhat), axis=[1, 2, 3]) / 2.
    
    prior_y = tf.reduce_sum(y * tf.math.log(1 / num_classes), axis=-1)
    
    pz = tf.reduce_sum(- 0.5 * tf.math.log(2 * np.math.pi) - 0.5 * (z ** 2), axis=-1)
    qz = tf.reduce_sum(- 0.5 * tf.math.log(2 * np.math.pi) - 0.5 * logvar  - 0.5 * ((z - mean) ** 2) / tf.math.exp(logvar), axis=-1)
    
    return error, prior_y, pz, qz
#%%