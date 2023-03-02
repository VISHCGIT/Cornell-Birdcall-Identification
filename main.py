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

sampling_rate = 22050 #Hz
read_file_upto = 4 #seconds 
seq_max_length = 10000 # sequence/padding length on data (without augmentation)
seq_max_length_aug = 2000 # sequence/padding length on data (with augmentation)

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

birdnames = ['aldfly', 'ameavo', 'amebit', 'amecro', 'amegfi', 'amekes',
       'amepip', 'amered', 'amerob', 'amewig', 'amewoo', 'amtspa',
       'annhum', 'astfly', 'baisan', 'baleag', 'balori', 'banswa',
       'barswa', 'bawwar', 'belkin1', 'belspa2', 'bewwre', 'bkbcuc',
       'bkbmag1', 'bkbwar', 'bkcchi', 'bkchum', 'bkhgro', 'bkpwar',
       'bktspa', 'blkpho', 'blugrb1', 'blujay', 'bnhcow', 'boboli',
       'bongul', 'brdowl', 'brebla', 'brespa', 'brncre', 'brnthr',
       'brthum', 'brwhaw', 'btbwar', 'btnwar', 'btywar', 'buffle',
       'buggna', 'buhvir', 'bulori', 'bushti', 'buwtea', 'buwwar',
       'cacwre', 'calgul', 'calqua', 'camwar', 'cangoo', 'canwar',
       'canwre', 'carwre', 'casfin', 'caster1', 'casvir', 'cedwax',
       'chispa', 'chiswi', 'chswar', 'chukar', 'clanut', 'cliswa',
       'comgol', 'comgra', 'comloo', 'commer', 'comnig', 'comrav',
       'comred', 'comter', 'comyel', 'coohaw', 'coshum', 'cowscj1',
       'daejun', 'doccor', 'dowwoo', 'dusfly', 'eargre', 'easblu',
       'easkin', 'easmea', 'easpho', 'eastow', 'eawpew', 'eucdov',
       'eursta', 'evegro', 'fiespa', 'fiscro', 'foxspa', 'gadwal',
       'gcrfin', 'gnttow', 'gnwtea', 'gockin', 'gocspa', 'goleag',
       'grbher3', 'grcfly', 'greegr', 'greroa', 'greyel', 'grhowl',
       'grnher', 'grtgra', 'grycat', 'gryfly', 'haiwoo', 'hamfly',
       'hergul', 'herthr', 'hoomer', 'hoowar', 'horgre', 'horlar',
       'houfin', 'houspa', 'houwre', 'indbun', 'juntit1', 'killde',
       'labwoo', 'larspa', 'lazbun', 'leabit', 'leafly', 'leasan',
       'lecthr', 'lesgol', 'lesnig', 'lesyel', 'lewwoo', 'linspa',
       'lobcur', 'lobdow', 'logshr', 'lotduc', 'louwat', 'macwar',
       'magwar', 'mallar3', 'marwre', 'merlin', 'moublu', 'mouchi',
       'moudov', 'norcar', 'norfli', 'norhar2', 'normoc', 'norpar',
       'norpin', 'norsho', 'norwat', 'nrwswa', 'nutwoo', 'olsfly',
       'orcwar', 'osprey', 'ovenbi1', 'palwar', 'pasfly', 'pecsan',
       'perfal', 'phaino', 'pibgre', 'pilwoo', 'pingro', 'pinjay',
       'pinsis', 'pinwar', 'plsvir', 'prawar', 'purfin', 'pygnut',
       'rebmer', 'rebnut', 'rebsap', 'rebwoo', 'redcro', 'redhea',
       'reevir1', 'renpha', 'reshaw', 'rethaw', 'rewbla', 'ribgul',
       'rinduc', 'robgro', 'rocpig', 'rocwre', 'rthhum', 'ruckin',
       'rudduc', 'rufgro', 'rufhum', 'rusbla', 'sagspa1', 'sagthr',
       'savspa', 'saypho', 'scatan', 'scoori', 'semplo', 'semsan',
       'sheowl', 'shshaw', 'snobun', 'snogoo', 'solsan', 'sonspa', 'sora',
       'sposan', 'spotow', 'stejay', 'swahaw', 'swaspa', 'swathr',
       'treswa', 'truswa', 'tuftit', 'tunswa', 'veery', 'vesspa',
       'vigswa', 'warvir', 'wesblu', 'wesgre', 'weskin', 'wesmea',
       'wessan', 'westan', 'wewpew', 'whbnut', 'whcspa', 'whfibi',
       'whtspa', 'whtswi', 'wilfly', 'wilsni1', 'wiltur', 'winwre3',
       'wlswar', 'wooduc', 'wooscj2', 'woothr', 'y00475', 'yebfly',
       'yebsap', 'yehbla', 'yelwar', 'yerwar', 'yetvir']
st.write(len(birdnames))
st.write(birdnames[0])
if uploaded_file is not None:
    st.write("file uploaded")
    #ipd.Audio(uploaded_file)
    start_sec = 0
    samples, sample_rate = librosa.load(uploaded_file, offset = start_sec, duration = 5)
    pred_labels = []
    for i in range (0, len(samples), 10000):
        samples_pad_seq = pad_seq_rawdata_on_1sample(samples[i: i+10000], 10000)
        samples_spectrogram = convert_spectrogram(samples_pad_seq)
        pred = best_model.predict(samples_spectrogram, verbose = 0)
        pred_labels.append(np.argmax(pred))
    pred_birds = set()
    for ele in pred_labels:
        pred_birds.add(birdnames[ele])

    st.write(len(samples))
