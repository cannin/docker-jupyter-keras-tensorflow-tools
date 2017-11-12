# From: https://raw.githubusercontent.com/hadim/keras-toolbox/master/kerastoolbox/visu.py

import keras.backend as K
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
import numpy as np

def make_mosaic(im, nrows, ncols, border=1):
    """From http://nbviewer.jupyter.org/github/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
    """

    nimgs = len(im)
    imshape = im[0].shape

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    im
    for i in range(nimgs):

        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = im[i]

    return mosaic


def plot_feature_map(model, layer_id, X, n=256, n_columns=3, ax=None, **kwargs):
    """
    """

    layer = model.layers[layer_id]

    try:
        get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
        activations = get_activations([X, 0])[0]
    except:
        # Ugly catch, a cleaner logic is welcome here.
        raise Exception("This layer cannot be plotted.")

    # For now we only handle feature map with 4 dimensions
    # For now we only handle feature map with 4 dimensions
    if activations.ndim != 4:
        raise Exception("Feature map of '{}' has {} dimensions which is not supported.".format(layer.name,
                                                                                             activations.ndim))

    # Set default matplotlib parameters
    if not 'interpolation' in kwargs.keys():
        kwargs['interpolation'] = "none"

    if not 'cmap' in kwargs.keys():
        kwargs['cmap'] = "binary"

    # Compute nrows and ncols for images
    n_mosaic = len(activations)
    nrows = n_mosaic // n_columns
    ncols = n_columns

    if ncols ** 2 < n_mosaic:
        nrows +=1

    # Compute nrows and ncols for mosaics
    if activations[0].shape[0] < n:
        n = activations[0].shape[0]

    nrows_inside_mosaic = int(np.round(np.sqrt(n)))
    ncols_inside_mosaic = int(nrows_inside_mosaic)

    if nrows_inside_mosaic ** 2 < n:
        ncols_inside_mosaic += 1

    fig_w = 15
    fig_h = nrows * fig_w / ncols

    fig = plt.figure(figsize=(fig_w, fig_h))

    for i, feature_map in enumerate(activations):

        # Without this transpose, activation map is a bunch of random vertical lines rather than something similar to the real image
        feature_map = np.transpose(feature_map, (2, 0, 1))

        mosaic = make_mosaic(feature_map[0:n], nrows_inside_mosaic, ncols_inside_mosaic, border=1)

        ax = fig.add_subplot(nrows, ncols, i+1)

        im = ax.imshow(mosaic, **kwargs)
        ax.set_title("Feature map #{} \nof layer#{} \ncalled '{}' \nof type {} ".format(i, layer_id,
                                                                                  layer.name,
                                                                                  layer.__class__.__name__))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        ax.axis('off')

        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)

    # fig.tight_layout()
    #return fig
    plt.show()

# Debugging code
# # This generates the same activations as in the script
# activations = get_activations(loaded_model, x_test[i:(i+1)], print_shape_only=True, layer_name='conv2d_1')
# activations = activations[0]
# activations = activations[0]
# print(activations.shape)
#
# # Without this transpose, activation map is a bunch of random vertical lines rather than something similar to the real image
# activations = np.transpose(activations, (2, 0, 1))
# print(activations.shape)
# plt.imshow(activations[0], interpolation='None', cmap='binary')