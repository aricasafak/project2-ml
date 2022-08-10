from flask import Flask, request
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
import cv2
import json

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def process(img):
    return cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3)
def predict(img):
    return model.layers[2](model.layers[1](model.layers[0](process(img)))).numpy()[0]

app = Flask(__name__)
model = keras.models.load_model('modelplant.hdf5', custom_objects={'FixedDropout':layers.Dropout})


@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    args = request.args
    img = load_image(args.get("imgName") + '.jpg')
    preds = predict(img)
    return "healthy: " + str(round(preds[0] * 100, 2)) + ", " + "scab: " + str(round(preds[1] * 100, 2)) + ", " + "rust: " + str(round(preds[2] * 100, 2)) + ", " + "multiple_diseases: " + str(round(preds[3] * 100, 2))

if __name__ == '__main__':
    app.run()

