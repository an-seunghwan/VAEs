#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import re

import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
#%%
with open("accuracy.txt", "w") as file:
    for model in ['pi', 'vat', 'mixmatch', 'plcb']: 
        dir = '/Users/anseunghwan/Documents/GitHub/semi/{}/logs/cifar10_4000'.format(model)
        file_list = [x for x in os.listdir(dir) if x not in ['.DS_Store', 'datasets', 'etc', 'warmup']]
        
        error = []
        for i in range(len(file_list)):
            path = dir + '/{}/test'.format(file_list[i])
            event_acc = EventAccumulator(path)
            event_acc.Reload()
            tag = 'accuracy'
            event_list = event_acc.Tensors(tag)
            value = tf.io.decode_raw(event_list[-1].tensor_proto.tensor_content, 
                                    event_list[-1].tensor_proto.dtype)
            error.append(100. * (1. - value.numpy()[0]))

        file.write("{} | test classification error | mean: {:.3f}, std: {:.3f}\n\n".format(model, np.mean(error), np.std(error)))
    
    for model in ['dgm', 'partedvae', 'shotvae']:
        dir = '/Users/anseunghwan/Documents/GitHub/semi/{}/logs/cifar10_4000'.format(model)
        model_list = [d for d in os.listdir(dir) if d != '.DS_Store']
        
        error = []
        inception = []
        for i in range(len(model_list)):
            with open(dir + '/' + model_list[i] + '/result.txt', 'r') as f:
                result = f.readlines()
            result = ' '.join(result) 
            
            """test classification error"""
            idx1 = re.search('TEST classification error: ', result).span()[1]
            idx2 = re.search('%', result).span()[0]
            error.append(float(result[idx1:idx2]))
            
            """Inception Score"""
            idx1 = re.search(' mean: ', result).span()[1]
            idx2 = re.search(', std: ', result).span()[0]
            inception.append(float(result[idx1:idx2]))
            
        file.write("{} | test classification error | mean: {:.3f}, std: {:.3f}\n".format(model, np.mean(error), np.std(error)))
        file.write("{} | Inception Score | mean: {:.3f}, std: {:.3f}\n\n".format(model, np.mean(inception), np.std(inception)))
#%%