# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:07:38 2021

@author: L
"""

import katan48x48 as ka
from keras.models import model_from_json
from os import urandom
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"#GPU selected for calculation

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
import time
start=time.perf_counter()

nr = 106
diff = 0xC0000845000
n=3000 

#load distinguishers

json_file = open('13194148204544model96.json','r');
json_model = json_file.read();

net = model_from_json(json_model);
net.load_weights('13194148204544best96depth.h5');


def wrong_key_decryption(n, diff, nr, net):
    means = np.zeros(2**16); sig = np.zeros(2**16);
    for i in range(2**16):
        keys=np.frombuffer(urandom(10*n),dtype=np.uint16).reshape(5,-1);
        keys = np.copy(keys);
        myKATAN = ka.KATAN(keys, 48, 254);
        all_key=myKATAN.change_key(keys);
        k_goal = 0x0^(all_key[194]<<15)^(all_key[196]<<14)^(all_key[198]<<13)^(all_key[199]<<12)^(all_key[200]<<11)^(all_key[201]<<10)^(all_key[202]<<9)^(all_key[203]<<8)^(all_key[204]<<7)^(all_key[205]<<6)^(all_key[206]<<5)^(all_key[207]<<4)^(all_key[208]<<3)^(all_key[209]<<2)^(all_key[210]<<1)^(all_key[211]);  
        k = i ^ k_goal
        X = ka.make_recover_data1(n, nr, diff, keys, k)
        Z = net.predict(X,batch_size=10000)
        Z = Z.flatten();
        means[i] = np.mean(Z);
        sig[i] = np.std(Z);
    return(means, sig);

means, sig = wrong_key_decryption(n, diff, nr, net)
#print(means)
np.save('data_wrong_key_mean_'+str(nr)+'r.npy', means)
np.save('data_wrong_key_std_'+str(nr)+'r.npy', sig)

end=time.perf_counter()   
print((end-start),'s')            
     
