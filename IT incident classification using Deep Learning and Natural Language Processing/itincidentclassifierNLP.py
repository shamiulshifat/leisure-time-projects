#IT incident classification using Deep Learning and Natural Language Processing

#reading libraries

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import re
import pickle as pk
import numpy as np
import keras as kr
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM,Embedding,Dropout,Bidirectional

#read dataset
pd.set_option('display.max_colwidth',-1) 
df= pd.read_excel("text") 
df= df.loc[:,"desc1":"Assignment group"]

#cleaning data
Cleaning the text using Regular Expressions

df["desc1"] = df["desc1"].fillna('').apply(str)
df["desc1"] = df["desc1"].str.lower()
df["desc1"] = df["desc1"].apply(lambda x: re.sub('http\S+:\/\/.*','',x))
df["desc1"] = df["desc1"].apply(lambda x: re.sub('\w*\d\w*"','',x))
df["desc1"] = df["desc1"].str.replace('[^\w\s]','')
df["desc1"] = df["desc1"].str.replace('\d+', '')

#Removing stop words

list2= ['words which you want to remove']
list1= list(stopwords.words('english'))
list3= list1+list2
df["desc1"]=df["desc1"].apply(lambda x: ' '.join([word for word in x.split() if word not in (list3)]))

#Removing un common words

freq=pd.Series(' '.join(df["desc1"]).split()).value_counts()[-3000:]

df["desc1"] = df["desc1"].apply(lambda x: ' '.join(x for x in x.split() if x not in freq))
df["desc1"] = df["desc1"].apply(lambda x:" ".join(x for x in x.split() if len(x) < 12))
df["desc1"] = df["desc1"].apply(lambda x:" ".join(x for x in x.split() if len(x) > 3))

#plot before tokenization
df['desc1'].str.len().plot.hist()

#Creating Word index and Tokenizer object

tokenizer = Tokenizer(num_words =8000,oov_token='<OOV>')
tokenizer.fit_on_texts(df["desc1"])
word_index = tokenizer.word_index
dict1=dict(list(word_index.items()))

#Saving and loading Tokenizer object for future encodings
#saving
with open('tokenizer.pickle', 'wb') as handle:
    pk.dump(tokenizer, handle, protocol=pk.HIGHEST_PROTOCOL)
#loading    
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pk.load(handle)
    
word_index = tokenizer.word_index
dict1=dict(list(word_index.items()))
print(dict1)

#Converting text into sequnece using Tokenizer object

tokenizer.fit_on_texts(df["desc1"])
X = tokenizer.texts_to_sequences(df["desc1"])
X= pad_sequences(X, maxlen=1000, padding="post", truncating="post")

#Encoding lables into numeric labels

le = LabelEncoder()
y=le.fit_transform(df["Assignment group"])

#Encoding numeric labels into One Hot encoded form

lb=np.array([0,1,2,3,4,5,6,7,8,9])
lb=dict(zip(le.classes_, lb))
y=tf.keras.utils.to_categorical(y, num_classes=10)
print(y)

#build model
model2=Sequential()
model2.add(Embedding(8000,32,input_length=X.shape[1]))
model2.add(Dropout(0.3))
model2.add(LSTM(32,return_sequences=True,dropout=0.3,recurrent_dropout=0.2))
model2.add(LSTM(32,dropout=0.3,recurrent_dropout=0.2))
model2.add(Dense(3,activation='softmax'))
model2.summary()
#compile model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


#train test split

from sklearn.model_selection import train_test_split X_train, X_test, y_train, y_test = train_test_split(X, y,   test_size=.1,          shuffle=True,     random_state=6)

#train model
batch_size=32 epochs=7 model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,
verbose=2, validation_split=.3)

#evaluate model
results=model.evaluate(x_test, y_test)
print(results)


