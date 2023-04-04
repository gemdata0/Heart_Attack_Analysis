# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 20:10:57 2023

@author: USER
"""
import numpy as np
import pickle 
import streamlit as st

# loading the saved model 
loaded_model = pickle.load(open('C:/Users/USER/Desktop/projects/Kaggle/Heart Attack Analysis/trained_model.sav', 'rb'))

# creating a function for prediction 

def heart_attack_prediction(input_data):
    

    # changing the input_data to numpy array 
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance 
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'will not have a heart attack'
    else:
      return 'will have a heart attack'
  
    
def main():
    
    
    # giving a title 
    st.tittle('Heart Attack Prediction')
    
    
    # getting input data from the user 

    age = st.text_input('age')
    sex = st.text_input('sex')
    cp = st.text_input('cp')
    trtbps = st.text_input('trtbps')
    chol = st.text_input('chol')
    thalachh = st.text_input('thalachh')
    exng = st.text_input('exng')
    oldpeak = st.text_input('oldpeak')
    slp = st.text_input('slp')
    caa = st.text_input('caa')
    thall = st.text_input('thall')
                          
    # code for prediction     
    diagnosis = ''

    # creating a button for prediction 
    if st.button('Heart Attack Test Results'):
        diagnosis = heart_attack_prediction([age,sex,cp,trtbps,chol,thalachh,exng,oldpeak,slp,caa,thall])        

    st.success(diagnosis)


if __name__=='__main__':
    main()
       