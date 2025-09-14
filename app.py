import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #disable msg, TF_CPP_MIN_LOG_LEVEL controls TensorFlow log messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, url_for
import numpy as np

app = Flask(__name__)

# Upload folder setup
UPLOAD_FOLDER = 'static/uploads'
#os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model("C:/Users/Dipali/OneDrive/Desktop/Data science/Image_classify.keras")

# Categories
data_cat = [
    'apple','banana','beetroot','bell pepper','cabbage',
    'capsicum','carrot','cauliflower','chilli pepper','corn','cucumber',
    'eggplant','garlic','ginger','grapes','jalepeno','kiwi','lemon','lettuce',
    'mango','onion','orange','paprika','pear','peas','pineapple','pomegranate',
    'potato','raddish','soy beans','spinach','sweetcorn','sweetpotato',
    'tomato','turnip','watermelon'
]

# Image size
img_height = 180
img_width = 180

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    accuracy = None
    image_path = None

    if request.method == 'POST':
        # Check if file is present
        if 'image' not in request.files:
            return render_template('index.html', error="No file uploaded")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        if file:
            # Save file to uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Load and preprocess image
            image_load = tf.keras.utils.load_img(filepath, target_size=(img_height, img_width))
            img_arr = tf.keras.utils.img_to_array(image_load)
            img_bat = tf.expand_dims(img_arr, 0)

            # Predict
            predict = model.predict(img_bat)
            score = tf.nn.softmax(predict[0])

            prediction = data_cat[np.argmax(score)]
            accuracy = np.max(score) * 100

            # Get path for HTML
            image_path = url_for('static', filename='uploads/' + file.filename)

    return render_template('index.html', prediction=prediction, accuracy=accuracy, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
