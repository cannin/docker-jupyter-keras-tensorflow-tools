import keras.backend as K

# From: https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py
#
# Taken conceptually in part from:
# https://arxiv.org/pdf/1312.6034.pdf
# https://azure.github.io/learnAnalytics-DeepLearning-Azure/saliency-maps.html
def compile_saliency_function(model, activation_layer='conv2d_1'):
    input_image = model.input
    layer_output = model.get_layer(activation_layer).output
    max_output = K.max(layer_output, axis=1) # axis=1 for dense layer
    saliency = K.gradients(max_output, input_image)[0]
    return K.function([input_image, K.learning_phase()], [saliency])
