# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:57:32 2020

@author: alimaleki100
"""
import numpy as np
from keras.preprocessing import image
from keras.models import load_model



#Load the Trained Model

classifier = load_model('C:/Users/session1/Documents/GitHub/CNN/catVSdog/catVSdog_Model.h5')


#Load the Test Image
test_image = image.load_img('C:/Users/session1/Documents/GitHub/CNN/catVSdog/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)