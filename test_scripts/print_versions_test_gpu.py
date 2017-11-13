
# coding: utf-8

# In[1]:


from __future__ import print_function
import os 

import cv2
import tensorflow as tf
import keras
import numpy

# Get versions
print(cv2. __version__)
print(tf.__version__)
print(keras.__version__)
print(numpy.__version__)

# Print current directory 
print(os.getcwd()) 

# Test GPU; Error results if no GPU
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

# Creates a session with log_device_placement set to True
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op
print(sess.run(c))

