# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:14:44 2020

@author: Admin
"""


# Note: you may need to update your version of future
# sudo pip install -U future

#import os, sys


from keras.preprocessing.sequence import pad_sequences
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from keras.models import load_model
import keras.backend as K
import numpy as np
import pandas as pd
import requests, json 
import pickle

def custom_loss(y_true, y_pred):
  # both are of shape N x T x K
  mask = K.cast(y_true > 0, dtype='float32')
  out = mask * y_true * K.log(y_pred)
  return -K.sum(out) / K.sum(mask)


def acc(y_true, y_pred):
  # both are of shape N x T x K
  targ = K.argmax(y_true, axis=-1)
  pred = K.argmax(y_pred, axis=-1)
  correct = K.cast(K.equal(targ, pred), dtype='float32')

  # 0 is padding, don't include those
  mask = K.cast(K.greater(targ, 0), dtype='float32')
  n_correct = K.sum(mask * correct)
  n_total = K.sum(mask)
  return n_correct / n_total


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    #text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text

# Python program to find current 
# weather details of any city 
# using openweathermap api 

 
def get_weather_information(city_name):
    # Enter your API key here 
    api_key = "8c367c293ca346315bcaa50402975461"
      
    # base_url variable to store url 
    base_url = "http://api.openweathermap.org/data/2.5/weather?" 
         
    # complete_url variable to store 
    # complete url address 
    complete_url = base_url + "q=" + city_name + "&appid=" + api_key  
      
    # get method of requests module 
    # return response object 
    response = requests.get(complete_url) 
      
    # json method of response object  
    # convert json format data into 
    # python format data 
    x = response.json()
    print(x)
      
    # Now x contains list of nested dictionaries 
    # Check the value of "cod" key is equal to 
    # "404", means city is found otherwise, 
    # city is not found 
    if x["cod"] != "404": 
      
        # store the value of "main" 
        # key in variable y 
        y = x["main"] 
      
        # store the value corresponding 
        # to the "temp" key of y
        current_temperature = "Temperature is " + str(y["temp"]) + " Kelvin"
        
        
      
        # store the value corresponding 
        # to the "pressure" key of y 
        current_pressure = "Pressure is " + str(y["pressure"]) + " hPa" 
      
        # store the value corresponding 
        # to the "humidity" key of y 
        current_humidiy = "Humdity is" + str(y["humidity"]) + " degrees"
      
        # store the value of "weather" 
        # key in variable z 
        z = x["weather"]
      
        # store the value corresponding  
        # to the "description" key at  
        # the 0th index of z 
        weather_description = "Weather desciption is " + z[0]["description"] 
      
        res = []
        res.append(current_temperature)
        res.append(current_pressure)
        res.append(current_humidiy)
        res.append(weather_description)
        result = ' '.join(res)
        
    else: 
        result = " City Not Found "
    return result

def KNN_preprocess(test_string):
    cv = CountVectorizer(max_features = 500)
    corpus=[]
    with open("KNN_corpus.txt", "r") as f:
        for line in f:
            corpus.append((line.strip()))
            cv.fit(corpus)
    X = corpus  
    y = [i for i in range(0,826)]
    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    bowVect = cv.fit(X_train)
    stopword = nltk.corpus.stopwords.words('english')
    stopword.append('asked')
    stopword.append('asking')
    
    
    review = test_string
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopword)]
    review = ' '.join(review)
    Test_Query=[]
    Test_Query.append(review)
    bowTest = cv.transform(Test_Query)
    return bowTest

def knn_predictor(s):
    knn_from_joblib = joblib.load('knn.pkl') 
    s = KNN_preprocess(s)
    predict_type = knn_from_joblib.predict(s)
    return predict_type 

def decode_sequence(input_seq):
    #INITIALIZATIONs
    word=''
    value=0
    
    word2idx_inputs_dict = {}
    word2idx_outputs_dict = {}
    word2idx_inputs = pd.read_csv('word2idx_inputs.csv')
    word2idx_outputs = pd.read_csv('word2idx_outputs.csv',encoding='windows-1250')
    
    for index, row in word2idx_inputs.iterrows():
        word=row['word']
        value=row['value']
        word2idx_inputs_dict[word] = value  
    
    
    for index, row in word2idx_outputs.iterrows():
        word=row['word']
        value=row['value']
        word2idx_outputs_dict[word] = value  
    
    
    idx2word_questions = {}
    idx2word_answers = {}
    idx2word_questions_file = pd.read_csv('idx2word_questions.csv',encoding='windows-1250')
    idx2word_answers_file = pd.read_csv('idx2word_answers.csv',encoding='windows-1250')
    
    for index, row in idx2word_questions_file.iterrows():
        word=row['value']
        value=row['word']
        idx2word_questions[value] = word  
    
    
    for index, row in idx2word_answers_file.iterrows():
        value=row['value']
        word=row['word']
        idx2word_answers[value] = word  
    
    
    #LOADING MODELS
    
    
    encoder_model = load_model('op_encoder.h5')
    encoder_model.load_weights('op_encoder_weights.h5')
    
    decoder_model = load_model('op_decoder.h5')
    decoder_model.load_weights('op_decoder_weights.h5') 
        
        
        
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    print(states_value)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first character of target sequence with the start character.
    # NOTE: tokenizer lower-cases all words
    target_seq[0, 0] = word2idx_outputs_dict['<sos>']

    # if we get this we break
    eos = word2idx_outputs_dict['<eos>']

    # Create the translation
    output_sentence = []
    for _ in range(1, 114):
        output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value
                )
   

        # Get next word
        idx = np.argmax(output_tokens[0, 0, :])
    
        # End sentence of EOS
        if eos == idx:
          break
    
        word = ''
        if idx > 0:
          word = idx2word_answers[idx]
          #print(word)
          output_sentence.append(word)
    
        # Update the decoder input
        # which is just the word just generated
        target_seq[0, 0] = idx
    
        # Update states
        states_value = [h, c]
        # states_value = [h] # gru

    return ' '.join(output_sentence)


def get_answer(s_original):
    max_len_input=12
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer_inputs = pickle.load(handle)
    input_sequences = tokenizer_inputs.texts_to_sequences([s_original])
    input_sequences = pad_sequences(input_sequences, maxlen=max_len_input)
    answer = decode_sequence(input_sequences)
    return answer 