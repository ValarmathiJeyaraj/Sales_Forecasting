# -*- coding: utf-8 -*-
"""Expsmooth_Sales.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cYC2W6_2302Ct0RBcsRgR16_MlcMq7IS
"""

!pip install ngrok



!pip install streamlit

import pandas as pd
import streamlit as st
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#from PIL import Image

!pip install pillow

#page configuration
#image = Image.open('inno.png')
#st.image(image,width = 500)
st.title('Time Series Cement Sales Forecasting Using Streamlit')
uploaded_file = st.file_uploader(r'/content/ds.csv', type=['csv'])

st.write("Enter the forecasting period")
level = st.slider("Select the level", 1, 300)
st.text('Selected: {}'.format(level))
if uploaded_file is not None:     
    cement = pd.read_excel(uploaded_file)
    cement['month'] = cement['month'].apply(lambda x: x.strftime('%D-%M-%Y'))
    
    hwe_model_mul_add = ExponentialSmoothing(cement["sales"][:130], seasonal = "add", trend = "add", seasonal_periods = 12).fit()
    
    newdata_pred = hwe_model_mul_add.predict(len(cement['sales']), len(cement['sales'])-3+level)
    
    
    st.subheader("For exponential model")
    pred = pd.DataFrame(newdata_pred, columns=['sales'])
    st.write("Sales Forecast: ", pred)
    st.line_chart(pred)
st.success("Success")
st.title('Thank you for your visit')

!streamlit run Expsmooth_Sales.ipynb