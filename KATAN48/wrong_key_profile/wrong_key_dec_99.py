# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:47:44 2021

@author: deeplearning
"""


import katan48x48 as ka
from keras.models import model_from_json
from os import urandom
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"#GPU selected for calculation

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
import time
start=time.perf_counter()

nr = 99
diff = 0xC0000845000
n=3000 

#load distinguishers

json_file = open('13194148204544model65.json','r');
json_model = json_file.read();

net = model_from_json(json_model);
net.load_weights('13194148204544best65depth.h5');


def wrong_key_decryption(n, diff, nr, net):
    means = np.zeros(2**4); sig = np.zeros(2**4);
    for i in range(2**4):
        keys=np.frombuffer(urandom(10*n),dtype=np.uint16).reshape(5,-1);
        keys = np.copy(keys); 
        myKATAN = ka.KATAN(keys, 48, 254);
        all_key=myKATAN.change_key(keys);
        k_goal = 0x0^(all_key[180]<<15)^(all_key[182]<<14)^(all_key[184]<<13)^(all_key[185]<<12)^(all_key[186]<<11)^(all_key[187]<<10)^(all_key[188]<<9)^(all_key[189]<<8)^(all_key[190]<<7)^(all_key[191]<<6)^(all_key[192]<<5)^(all_key[193]<<4)^(all_key[194]<<3)^(all_key[195]<<2)^(all_key[196]<<1)^(all_key[197]);
        k = i ^ k_goal
        X = ka.make_recover_data2(n, nr, diff, keys, k)
        Z = net.predict(X,batch_size=10000)
        Z = Z.flatten();
        means[i] = np.mean(Z);
        sig[i] = np.std(Z);
    return(means, sig);

means, sig = wrong_key_decryption(n, diff, nr, net)
print(means)
#np.save('data_wrong_key_mean_'+str(nr)+'r.npy', means)
#np.save('data_wrong_key_std_'+str(nr)+'r.npy', sig)

end=time.perf_counter()   
print((end-start),'s')            
     
