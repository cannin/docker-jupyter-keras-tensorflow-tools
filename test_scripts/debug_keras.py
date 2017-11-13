
# coding: utf-8

# # Debug Convolutional Neural Networks in Keras
# 
# Debug Keras models using the following: 
# 
# * Tensorflow debugger: tfdbg 
# * Model memory usage
# * Visualize model 
# * Visualize CNN layers 

# In[ ]:


from time import time

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

from keras.callbacks import TensorBoard

from keras.datasets import imdb

import keras.backend as K
from tensorflow.python import debug as tf_debug
import tensorflow as tf

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import os

import resource 


# In[10]:


# Trigger tfdbg
# NOTE: https://www.tensorflow.org/programmers_guide/debugger#debugging_keras_models_with_tfdbg
#K.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))


# In[ ]:


# Print memory usage of this current Python process 
import resource; print(round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2), 'MB')


# In[10]:


max_features = 20000
maxlen = 80
batch_size = 32

print(os.getcwd())

tensorboard = TensorBoard(log_dir="tf_logs/{}".format(time()))


# In[11]:


#from keras.datasets import cifar10
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[16]:


#plot_model(model, to_file='model.png')
SVG(model_to_dot(model, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))


# In[8]:


model.fit(x_train, y_train, batch_size=batch_size, epochs=2, validation_data=(x_test, y_test), verbose=1, callbacks=[tensorboard]) 


# In[8]:


score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

