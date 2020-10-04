# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:12:10 2020

@author: Admin
"""

#Import Libraries
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Model
#Script to convert data to lower case and save back to csv
"""
df = pd.read_csv('Pune.csv')
Question = df['Question']
Answer = df['Answer']
Cat = df['Category']

i=0
for word in df['Category']:
    if (isinstance(word, str)):
        word = word.lower()
        Cat[i] = word
    i=i+1
   
i=0
for word in df['Question']:
    if (isinstance(word, str)):
        word = word.lower()
        Question[i] = word
    i=i+1
    
i=0
for word in df['Answer']:
    if (isinstance(word, str)):    
        word = word.lower()
        Answer[i] = word
    i=i+1

c_arr = np.array(Cat)
a_arr = np.array(Answer)
q_arr = np.array(Question)
    
d={'Category': [i for i  in c_arr], 'Question': [i for i in q_arr],'Answer': [i for i in a_arr]}
 
dataset = pd.DataFrame(data=d,columns=['Category', 'Question', 'Answer'])
df.to_csv('pre_processed_Pune.csv', header=False, index=False) 

"""
#Import Dataset
df = pd.read_csv('pre_processed_Nashik.csv')


#Data Cleaning
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

stopword = nltk.corpus.stopwords.words('english')
stopword.append('asked')
stopword.append('asking')

for review in df['Question']:
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopword)]
    review = ' '.join(review)
    corpus.append(review)
    
with open("KNN_corpus.txt", "w") as f:
    for s in corpus:
        f.write(str(s) +"\n")


    
#Creating Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 500)

X = corpus
y = df.iloc[:,0]
    
#Splitting data to training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=0)


bowVect = cv.fit(X_train)
bowTrain = bowVect.transform(X_train)
bowTest = bowVect.transform(X_test)

#Create Classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(bowTrain, Y_train)

from sklearn.externals import joblib 

# Save the model as a pickle in a file 
joblib.dump(classifier , 'knn.pkl') 

#Predict Results
y_pred = classifier.predict(bowTest)

#Evaluating the results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

from sklearn.metrics import accuracy_score
score = accuracy_score(Y_test,y_pred)
score*100

#Testing the Classifier
test_string = 'What is the weather today?'
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

# Load the model from the file 
knn_from_joblib = joblib.load('knn.pkl') 

# Use the loaded model to make predictions 
prediction = knn_from_joblib.predict(bowTest) 
print(prediction)
