# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:34:04 2022

@author: Lenovo
"""


from keras.models import model_from_json

json_file = open('13194148204544model96.json','r');
json_model = json_file.read();
net50 = model_from_json(json_model);
net50.load_weights('13194148204544best96depth.h5');
net50.summary()