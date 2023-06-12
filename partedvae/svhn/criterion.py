#%%
import numpy as np
import tensorflow as tf
#%%
def ELBO_criterion(xhat, image, z_mean, z_logvar, c_logit, u_mean, u_logvar, model, num_classes, args):
    '''reconstruction'''
    if args['bce_reconstruction']:
        recon_loss = - tf.reduce_mean(tf.reduce_sum(image * tf.math.log(tf.clip_by_value(xhat, 1e-10, 1.)) + 
                                                    (1. - image) * tf.math.log(1. - tf.clip_by_value(xhat, 1e-10, 1.)), axis=[1, 2, 3]))
    else:
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.math.abs(image - xhat), axis=[1, 2, 3]))
            
    '''KL-divergence of z'''
    z_kl = tf.reduce_mean(tf.reduce_sum(- 0.5 * (1 + z_logvar - tf.math.pow(z_mean, 2) - tf.math.exp(z_logvar)), axis=-1))
    
    '''KL-divergence of c (marginal)'''
    qc_x = tf.nn.softmax(c_logit, axis=-1)
    qc = tf.reduce_mean(qc_x, axis=0)
    agg_c_kl = tf.reduce_sum(qc * (tf.math.log(tf.clip_by_value(qc, 1e-10, 1.)) - tf.math.log(1. / num_classes)))
    
    '''entropy of c'''
    c_entropy = tf.reduce_mean(- tf.reduce_sum(qc_x * tf.math.log(tf.clip_by_value(qc_x, 1e-10, 1.)), axis=-1))
    
    '''mixture KL-divergence of u'''
    u_means = tf.tile(u_mean[:, tf.newaxis, :], (1, num_classes, 1))
    u_logvars = tf.tile(u_logvar[:, tf.newaxis, :], (1, num_classes, 1))
    u_kl = tf.reduce_sum(0.5 * (tf.math.pow(u_means - model.u_prior_means, 2) / tf.math.exp(model.u_prior_logvars)
                                - 1
                                + tf.math.exp(u_logvars) / tf.math.exp(model.u_prior_logvars)
                                + model.u_prior_logvars
                                - u_logvars), axis=-1)
    u_kl = tf.reduce_mean(tf.reduce_sum(tf.multiply(qc_x, u_kl), axis=-1))
    
    '''Bhattacharyya coefficient'''
    u_var = tf.math.exp(model.u_prior_logvars)
    avg_u_var = 0.5 * (u_var[tf.newaxis, ...] + u_var[:, tf.newaxis, :])
    inv_avg_u_var = 1. / (avg_u_var + 1e-8)
    diff_mean = model.u_prior_means[tf.newaxis, ...] - model.u_prior_means[:, tf.newaxis, :]
    D = 1/8 * tf.reduce_sum(diff_mean * inv_avg_u_var * diff_mean, axis=-1)
    D += 0.5 * tf.reduce_sum(tf.math.log(avg_u_var + 1e-8), axis=-1)
    D += - 0.25 * (tf.reduce_sum(model.u_prior_logvars, axis=-1)[tf.newaxis, ...] + tf.reduce_sum(model.u_prior_logvars, axis=-1)[:, tf.newaxis])
    BC = tf.math.exp(- D)
    
    return recon_loss, z_kl, agg_c_kl, c_entropy, u_kl, BC
#%%