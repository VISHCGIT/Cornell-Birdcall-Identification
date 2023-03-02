import streamlit as st
#import ipython.display as ipd 
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import  LabelEncoder
#from miscfunctions import get_best_model
import numpy as np
import re
import datetime
import tensorflow as tf
import tensorflow as tf
import keras
from keras import layers
from keras import applications
from keras.regularizers import l2, l1
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils, to_categorical
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Conv2D, LSTM, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras.optimizers import Adam
from keras.optimizers import SGD
import librosa
#import librosa.display
st.header("Rossmann Sales Prediction App")

# Called while training, to calculate F1 Score with average 'micro'
def micro_f1(y_true, y_pred):
    try:
        idx = np.argmax(tf.Variable(tf.abs(y_pred)), axis=1) # getting index of maximum value in each row
        val_pred_label = tf.Variable(tf.zeros_like(y_pred))
        for i in range(val_pred_label.shape[0]):
            val_pred_label = val_pred_label[i, idx[i]].assign(1)
        micro_f1 = f1_score(y_true, val_pred_label, average='micro')
    except ValueError:
        pass
    return micro_f1

# load model
tf.keras.backend.clear_session()
input_shape = (64, 20, 1)
st.write("input_shape: ",input_shape)
def get_best_model(input_shape, num_classes):
    model=Sequential()

    model.add(Conv2D(32, (3,3),activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(2))

    model.add(Conv2D(64, (3,3), activation="relu"))   # extra pad
    model.add(MaxPooling2D((2,2)))                    #extra pad

    model.add(Conv2D(128, (3,3), activation="relu"))  # extra pad
    model.add(MaxPooling2D((2,2), padding='same'))

    model.add(Conv2D(128, (3,3), activation="relu", padding='same'))
    model.add(MaxPooling2D((2,2), padding='same'))

    model.add(BatchNormalization())#act
    model.add(Flatten())
    model.add(Dropout(0.2)) #act
    model.add(Dense(512,activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    #model.summary()

    opt = optimizers.SGD(lr=1e-4, momentum=0.9)
    model.compile(loss ='binary_crossentropy', 
                  optimizer = 'adam',
                  metrics = ['accuracy', micro_f1])
    return model
best_model = get_best_model(input_shape, 264)
st.write(best_model.summary())
best_model.run_eagerly = True
st.write("after run eagerly")
best_model.load_weights("best_model_cnn_spect.hdf5")
st.write("after model load");

def pad_seq_rawdata_on_1sample(audio_sample, seq_max_length):
    """This function takes rawdata (array 2x2) and trucate or pad based on input seq_max_length"""
    try: 
        rawdata_pad_seq = []
        audio_sample = list(audio_sample)
        if len(audio_sample) > seq_max_length:
            pad_seq = audio_sample[0:seq_max_length]
        else:
            pad_seq = list(audio_sample + [0]*(seq_max_length-len(audio_sample)))
        rawdata_pad_seq.append(pad_seq)
        rawdata_pad_seq = np.array(rawdata_pad_seq)
        return rawdata_pad_seq
    except:
        print("Something went wrong")
    finally:
        del rawdata_pad_seq
        gc.collect()

        
def mel_spectrogram(rawdata, sr=22050, mels=64):
    '''Converting to spectrogram'''
    spectrum = librosa.feature.melspectrogram(y=rawdata, sr=sr, n_mels=mels)
    logmel_spectrum = librosa.power_to_db(S=spectrum, ref=np.max)
    return logmel_spectrum
        
def convert_spectrogram(data_pad_seq):
    try:
        spectrogram = []
        for i in range(len(data_pad_seq)):
            spectrogram.append(mel_spectrogram(data_pad_seq[i]))
            #if i%10 == 0:
            #    print("Processed till: ",i)
        spectrogram = np.array(spectrogram)
        return spectrogram
    except:
        print("Something went wrong",e)
    finally:
        del spectrogram
        gc.collect
        
uploaded_file = st.file_uploader("Choose a file")
st.write("before upload123")

if uploaded_file is not None:
    st.write("file uploaded")
    #ipd.Audio(uploaded_file)
    start_sec = 0
    samples, sample_rate = librosa.load(uploaded_file, offset = start_sec, duration = 5)
    st.write(len(samples))
