

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import json
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add




def preprocess_image(img):
    
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    
    img = preprocess_input(img)
    
    return img




def encode_img(img):
    img = preprocess_image(img)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((1,-1))
    return feature_vector



def predict_caption(photo):
    max_len = 35
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() 
        word = idx_to_word[ypred]
        in_text += (' ' + word)
        
        if word == "endseq":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption



model = load_model('model_19.h5')
model.make_predict_function()


model_temp = ResNet50(weights='imagenet',input_shape=(224,224,3))

model_new = Model(model_temp.input,model_temp.layers[-2].output)
model_new.make_predict_function()


with open('w2i.pkl','rb') as f:
    word_to_idx = pickle.load(f)
    
    
with open('i2w.pkl','rb') as f:
    idx_to_word = pickle.load(f)





def caption(image):
    img_enc = encode_img(image)
    caption = predict_caption(img_enc)
    
    return caption



