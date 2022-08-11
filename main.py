from flask import Flask, request
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import urllib.request

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def process(img):
    return cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3)
def predict(img):
    return model.layers[2](model.layers[1](model.layers[0](process(img)))).numpy()[0]

app = Flask(__name__)
model = keras.models.load_model('modelplant.h5', custom_objects={'FixedDropout':layers.Dropout})


@app.route('/predict', methods=['GET', 'POST'])
def welcome():
    args = request.args
    link = args.get("imgName")
    urllib.request.urlretrieve("https://storage.googleapis.com/download/storage/v1/b/leaf-ba4be.appspot.com/o/images%2F" + link + ".jpg?alt=media", "temp.jpg")
    #urllib.request.urlretrieve("https://storage.googleapis.com/download/storage/v1/b/leaf-ba4be.appspot.com/o/images%2Fimage.jpg?&alt=media", "temp.jpg")
    img = load_image('temp.jpg')
    preds = predict(img)
    return "healthy: " + str(round(preds[0] * 100, 2)) + ", " + "scab: " + str(round(preds[1] * 100, 2)) + ", " + "rust: " + str(round(preds[2] * 100, 2)) + ", " + "multiple_diseases: " + str(round(preds[3] * 100, 2))



if __name__ == '__main__':
    app.run()

