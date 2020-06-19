# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:15:18 2020

@author: Ali
"""

import numpy as np

# Flask
from flask import Flask,  request, render_template
import os

from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img


app=Flask(__name__)

app.config["IMAGE_UPLOADS"] = "C:/Users/Ali/Desktop/FlaskCNNProject/uploads"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]

classifier = load_model('e:/catVSdog_Model.h5')

def predict(iimage, target):
    iimage=iimage.resize(target)
    iimage = img_to_array(iimage)
    iimage = np.expand_dims(iimage, axis = 0)
    prediction = classifier.predict(iimage)
        
    if prediction[0][0] == 1:
        result = 'dog'
    else:
        result = 'cat'
    return result


@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            uploaded_image = request.files["image"]
            uploaded_image.save(os.path.join(app.config["IMAGE_UPLOADS"], uploaded_image.filename))
            
            test_image = load_img("C:/Users/Ali/Desktop/FlaskCNNProject/uploads/"+uploaded_image.filename, target_size = (64, 64))
            #uploaded_image = Image.open(io.BytesIO(uploaded_image))
            result=predict(test_image, target=(64, 64))
            uploaded_image.save(os.path.join(app.config["IMAGE_UPLOADS"], uploaded_image.filename))


            return render_template('index.html', result = result)

            #return redirect(request.url)


    return render_template("index.html")







if __name__== '__main__':
	app.run()	

