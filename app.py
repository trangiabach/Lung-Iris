from flask import Flask, render_template, request, redirect, flash, url_for
from flask.json import jsonify
from werkzeug.utils import secure_filename
import os

from PIL import Image 
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow import keras

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

IMG_SIZE = 256

model_path = 'CNN-COVID-verOfficial-98.h5'

model = tf.keras.models.load_model(model_path)


def preprocess(image):
  img = cv2.imread(image, 0)
  img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
  ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
  thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2RGB) * 255
  thresh1 = np.expand_dims(thresh1, axis=0)
  return thresh1

def pneumonia_predict(img):
    image = preprocess(img)

    classes = model.predict(image)
    if classes[0][0] > 0.5:
        prediction = 'PNEUMONIA'
    else:
        prediction = 'NORMAL'
    return [prediction, str(round(classes[0][0], 2)*100), str(round(1 - classes[0][0], 2)*100), str(classes[0][0])] 


app = Flask(__name__)

UPLOAD_FOLDER = 'static'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def append_id(filename):
    return "{0}_{2}.{1}".format(*filename.rsplit('.', 1) + '-heatmap')

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    return superimposed_img



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def submit_file():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')

        file_paths = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                label = pneumonia_predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                img = preprocess(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                model.layers[-1].activation = None
                print(model.summary())
                heatmap = make_gradcam_heatmap(img, model, 'conv2d_83')
                heatmap_img = save_and_display_gradcam(os.path.join(app.config['UPLOAD_FOLDER'], filename),heatmap)
                heatmap_img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                file_paths.append({
                    'path' : os.path.join(app.config['UPLOAD_FOLDER'], filename),
                    'prediction': label[0],
                    'pneumonia-percentage': label[1],
                    'normal-percentage': label[2],
                    'a': label[3]
                })
        flash('File(s) successfully uploaded')
    return jsonify({
        'list': file_paths
    })




if __name__ == '__main__':
    app.run(debug= True)
