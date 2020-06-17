import os
from flask import Flask
from flask import request
from flask import render_template

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#import tensorflow as tf
#from keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
UPLOAD_FOLDER = "/Users/user/Desktop/web_app/static"
DEVICE = "cuda"
MODEL = None



def predict(image_location):
    classifier = Sequential()

    classifier.add(Convolution2D(32, 3, 3, input_shape =(64, 64, 3), activation= 'relu'))

    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Flatten())

    classifier.add(Dense(output_dim = 128, activation = 'relu'))
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory('data/train', target_size=(64, 64), batch_size=32, class_mode='binary')

    test_set = test_datagen.flow_from_directory('static', target_size=(64, 64), batch_size=32, class_mode='binary')
 
    history = classifier.fit_generator(training_set, samples_per_epoch=800, nb_epoch=1, validation_data=test_set, nb_val_samples=200)

    return history.history['accuracy']

@app.route("/", methods =["GET", "POST"])

def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )

            image_file.save(image_location)
            pred = predict(image_location)
            return render_template("index.html",prediction=pred)
    return render_template("index.html",prediction=0)

if __name__ == "__main__":
    app.run(port=1200, debug=True)
