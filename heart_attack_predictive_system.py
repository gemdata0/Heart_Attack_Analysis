# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 19:30:04 2023

@author: USER
"""

import numpy as np 
import pickle


# loading the saved model 
loaded_model = pickle.load(open('C:/Users/USER/Desktop/projects/Kaggle/Heart Attack Analysis/trained_model.sav', 'rb'))

input_data = (63,1,3,145,233,150,0,2.3,0,0,1)

# changing the input_data to numpy array 
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('will not have a heart attack')
else:
  print('will have a heart attack')