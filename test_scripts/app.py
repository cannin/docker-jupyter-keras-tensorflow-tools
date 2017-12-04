from flask import Flask, request, jsonify

import os

# NOTE: Add allowed extensions http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

UPLOAD_FOLDER = os.path.basename('uploads')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# LOAD MODEL
def load_model():
    from keras.applications.vgg16 import VGG16
    import tensorflow as tf

    model = VGG16(weights='imagenet', include_top=True)
    graph = tf.get_default_graph()

    return model, graph


model, graph = load_model()

# SET UP API
@app.route('/predict', methods=['POST'])
def predict():
    from keras.preprocessing import image
    from keras.applications.vgg16 import preprocess_input
    import numpy as np

    file = request.files['file']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    with graph.as_default():
        out = model.predict(x)

        print("OUT: ", out)
        print("AM: ", np.argmax(out, axis=1))

        response = int(np.argmax(out, axis=1))

        return jsonify(response)


if __name__ == "__main__":
    app.run()
