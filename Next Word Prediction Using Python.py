#!/usr/bin/env python
# coding: utf-8

# # Next Word Prediction Model using Python
# 
# Next Word Prediction means predicting the most likely word or phrase that will come next in a sentence or text. It is like having an inbuilt feature on an application that suggests the next word as you type or speak. The Next Word Prediction Models are used in applications like messaging apps, search engines, virtual assistants, and autocorrect features on smartphones. So, if you want to learn how to build a Next Word Prediction Model, this article is for you. In this article, I’ll take you through building a Next Word Prediction Model with Deep Learning using Python.
# 
# #To build a Next Word Prediction model:
# 
# 1.start by collecting a diverse dataset of text documents, 
# 2.preprocess the data by cleaning and tokenizing it, 
# 3.prepare the data by creating input-output pairs, 
# 4.engineer features such as word embeddings, 
# 5.select an appropriate model like an LSTM or GPT, 
# 6.train the model on the dataset while adjusting hyperparameters,
# 7.improve the model by experimenting with different techniques and architectures.

# In[3]:


#import liberies
import numpy as np 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


# In[4]:


#Read the text file 
with open(r"D:\Data science\Data\sherlock-holm.es_stories_plain-text_advs.txt",'r', encoding='utf-8') as file:
    text = file.read()


# In[5]:


text


# In[6]:


#Now let’s tokenize the text to create a sequence of words:
tokenizer=Tokenizer()
tokenizer.fit_on_texts([text])
total_words=len(tokenizer.word_index) + 1


# In[7]:


#Now let’s create input-output pairs by splitting the text into sequences of tokens and forming n-grams from the sequences:
input_sequences = []
for line in text.split('\n'):
    token_list=tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


# In[8]:


#Now let’s pad the input sequences to have equal length:
max_sequence_len= max([len(seq) for seq in input_sequences])
input_sequences=np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))


# In[9]:


input_sequences


# In[10]:


max_sequence_len


# In[11]:


#Now let’s split the sequences into input and output
X=input_sequences[:, :-1]
y=input_sequences[:, -1]


# In[12]:


X


# In[13]:


y


# In[14]:


#Now let’s convert the output to one-hot encode vectors:
y=np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))


# In[15]:


y


# In[16]:


#Now let’s build a neural network architecture to train the model:
model=Sequential()
model.add(Embedding(total_words, 100,input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation="softmax"))
print(model.summary())


# In[17]:


#Now let’s compile and train the model:
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(X,y,epochs=10, verbose=1)


# In[18]:


#we can generate the next word predictions using our model:
seed_text = "I will leave if they"
next_words = 3

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)


# In[ ]:




