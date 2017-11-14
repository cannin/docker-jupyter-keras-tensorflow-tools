
# coding: utf-8

# In[1]:


from __future__ import print_function

import os 
import sys
import subprocess

import cv2
import tensorflow as tf
import keras as K
import numpy as np

from utils.get_gpu_name import * 

from tensorflow.python.client import device_lib


# In[2]:


# Get versions
print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Keras: ", K.__version__)
print("Numpy: ", np.__version__)
print("Tensorflow: ", tf.__version__)
print("Keras Backend: ", K.backend.backend())
print("GPU: ", get_gpu_name())

# Print current directory 
print(os.getcwd()) 

print(device_lib.list_local_devices())

# Test GPU; Error results if no GPU
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

# Creates a session with log_device_placement set to True
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op
print(sess.run(c))

