import streamlit as st
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import  LabelEncoder
#from miscfunctions import get_best_model
import numpy as np
import re
import datetime
import tensorflow as tf
st.header("Rossmann Sales Prediction App")

# load model
tf.keras.backend.clear_session()
input_shape = (64, 20, 1)
st.write("input_shape: ",input_shape)
#best_model = get_best_model(input_shape, 264)
#best_model.run_eagerly = True
#best_model.load_model("best_model_cnn_spect.hdf5")
