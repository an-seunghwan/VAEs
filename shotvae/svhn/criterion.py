#%%
import tensorflow as tf
#%%
def ELBO_criterion(args, num_classes, x, xhat, mean, log_sigma, log_prob):
    '''reconstruction'''
    recon_loss = tf.reduce_mean(- tf.reduce_sum(x * tf.math.log(xhat) + 
                                                (1. - x) * tf.math.log(1. - xhat), axis=[1, 2, 3]))
    # if args['br']:
    #     recon_loss = tf.reduce_mean(- tf.reduce_sum(x * tf.math.log(xhat) + 
    #                                                 (1. - x) * tf.math.log(1. - xhat), axis=[1, 2, 3]))
    # else:
    #     recon_loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(xhat - x) / (2. * (args['x_sigma'] ** 2)), axis=[1, 2, 3]))
        
    '''KL-divergence'''
    kl_z = tf.reduce_mean(0.5 * tf.reduce_sum(tf.math.square(mean) + tf.math.exp(2. * log_sigma) - (2. * log_sigma) - 1, axis=-1))
    kl_y = tf.reduce_mean(tf.reduce_sum(tf.math.exp(log_prob) * (log_prob - tf.math.log(1. / num_classes)), axis=1))
    return recon_loss, kl_z, kl_y
#%%