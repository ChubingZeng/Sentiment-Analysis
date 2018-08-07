#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 21:39:10 2018

@author: chubingzeng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:39:25 2018

@author: chubingzeng
"""

# =============================================================================
# install packages
# =============================================================================
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
%matplotlib inline
from subprocess import check_output
import string
import tensorflow as tf
import keras
from ml_utils import *
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn import metrics

import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Embedding
from keras.preprocessing import sequence
import emoji
# =============================================================================
# read in data
# =============================================================================
reviews = pd.read_csv("reviews.csv", sep='|')
reviews.head()

train = reviews[reviews.index % 5 != 0]  # Excludes every 3rd row starting from 0
test = reviews[reviews.index % 5 == 0]
#train.shape
#test.shape

train_x, train_y = train['text'].tolist(), train['label'].tolist()
test_x, test_y = test['text'].tolist(), test['label'].tolist()

print('training size:', len(train_x))
print('test size:', len(test_x))

# =============================================================================
# Preprocessing
# =============================================================================
lemmatizer = nltk.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
transtbl = str.maketrans(string.punctuation, ' ' * len(string.punctuation)) # remove blamk space

def preprocessing(line: str) -> str:
    line = line.replace('<br />', '').translate(transtbl)
    
    tokens = [lemmatizer.lemmatize(t.lower(),'v')  # What to put in the list
              for t in nltk.word_tokenize(line)]    # Where 
#              if t.lower() not in stopwords]       # If
    
    return ' '.join(tokens)

train_x = list(map(preprocessing, train_x))
test_x = list(map(preprocessing, test_x))

all_words = []
word_train = []
word_test = []
for review in train_x:
    word_train.append(review.lower().split())
    for word in review.split():
        all_words.append(word.lower())

for review in test_x:
    word_test.append(review.lower().split())
    for word in review.split():
        all_words.append(word.lower())

counter = Counter(all_words)
vocab = sorted(counter, key=counter.get, reverse=True)

vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}

# Step 1.6 Using vocab_to_int to transform each review to vector of numbers
train_to_ints = []
for review in word_train:
    train_to_ints.append([vocab_to_int[word] for word in review])

test_to_ints = []
for review in word_test:
    test_to_ints.append([vocab_to_int[word] for word in review])

reviews_lens = Counter([len(x) for x in train_to_ints])
print('Zero-length {}'.format(reviews_lens[0]))
print("Max review length {}".format(max(reviews_lens)))


seq_len = 250

features = np.zeros((len(train_to_ints), seq_len), dtype=int)
for i, review in enumerate(train_to_ints):
    features[i, -len(review):] = np.array(review)[:seq_len]
    
features_test = np.zeros((len(test_to_ints), seq_len), dtype=int)
for i, review in enumerate(test_to_ints):
    features_test[i, -len(review):] = np.array(review)[:seq_len]


def convert_01(X):
    if X == "positive":
        return 1
    if X == "negative":
        return 0

X_train = features
y_train = np.array(list(map(convert_01, train_y)))

X_test = features_test
y_test = np.array(list(map(convert_01, test_y)))

# =============================================================================
# Done with preprocessing pipeline
# =============================================================================


# =============================================================================
# LSTM Modeling
# =============================================================================

# Pad sequences
x_train = sequence.pad_sequences(X_train, maxlen=200)
x_test = sequence.pad_sequences(X_test, maxlen=200)
x_train.shape
#x_train = x_train[:25000]
#y_train = y_train[:25000]
model = Sequential()
model.add(Embedding(len(vocab_to_int) + 1, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

batch_size = 600
history = model.fit(x_train, 
          y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(x_test, y_test),
          shuffle=True)

x_train.shape
# Save model structure
with open("amazon_model.json", "w") as fp:
    fp.write(model.to_json())

# Save model weights
model.save_weights("amazon_model.h5")

1
h = history.history.copy()


plt.plot(h['acc'])
plt.plot(h['val_acc'])

plt.plot(h['loss'])
plt.plot(h['val_loss'])



# Predict
pred = model.predict_classes(x_test)
print(pred)

# Metrics
# metrics.accuracy_score(y_true, y_pred)
from sklearn import metrics
accuracy = metrics.accuracy_score(pred,y_test)
print(accuracy)

# Use keyword arguments to set arguments explicitly
print(metrics.classification_report(y_true=y_test, y_pred=pred))

metrics.roc_curve(y_test,pred)

#The list of lists
# Compute ROC curve and ROC area for each class
from sklearn.metrics import roc_curve, auc
n_classes = 2
y_pred = model.predict_proba(x_test)
y_pred_list = np.array([y for x in y_pred for y in x])


fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_list)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)'% roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],color='navy', lw=lw, linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()






x_new = np.array(['Easy on your skin and removes make up easily.: I first found these at Trader Joes and picked them up on a whim. I was pleasantly surprised at how well they removed my makeup and left my skin feeling clean and cared for and have often followed with moisturizer after a long day at work so I could at least get my makeup off before I went to bed. Of course for days that I have more time and energy I can follow with my normal cleansing routine.Ive tried a few other brands but have always returned to Comodynes as the other brands would start to irritate my skin even in a weeks time and Ive never experienced any irritation with Comodynes.I ordered the 3-pack this time but will order the 6-pack next time for the value and would definitely buy it through the Subscribe & Save program if it were offered.'])
x_new = list(map(preprocessing, x_new))

word_new = []
for review in x_new:
    word_new.append(review.lower().split())

new_to_ints = []
for review in word_new:
    for word in review:
        if word not in vocab_to_int.keys():
            new_to_ints.append(0)
        else:
            new_to_ints.append(vocab_to_int[word])
new_to_ints = [new_to_ints]
features_new = np.zeros((1, seq_len), dtype=int)
for i, review in enumerate(new_to_ints):
    features_new[i, -len(review):] = np.array(review)[:seq_len]




X_test_indices = sequence.pad_sequences(features_new, maxlen=180)
len(model.predict(x_train[0]))
len(x_train[0])

model.predict_classes(X_test_indices)
model.predict_classes(x_train[0:2])

