from flask import Flask, request
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    args = request.args

    label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    model = keras.models.load_model('model_optimal.h5')

    test_img = image.load_img(args.get("imagePath")+'.jpg', target_size=(224, 224), color_mode="grayscale")
    test_img = np.expand_dims(test_img, axis=0)
    test_img = test_img.reshape(1, 224, 224, 1)
    result = model.predict(test_img)

    result = list(result[0])
    img_index = result.index(max(result))
    return label_dict[img_index]
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)

