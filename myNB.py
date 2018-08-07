#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 21:59:19 2018

@author: chubingzeng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 12:52:33 2018

@author: chubingzeng
"""

#### install packages ####
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

# =============================================================================
# read in data
# =============================================================================
reviews = pd.read_csv("reviews.csv", sep='|')
reviews.head()

#all_labels = reviews.iloc[:, 0]
#np.unique(all_labels)

train = reviews[reviews.index % 5 != 0]  # Excludes every 3rd row starting from 0
test = reviews[reviews.index % 5 == 0]
train.shape
test.shape


# =============================================================================
# Load data
# =============================================================================


def load_data(path):
    data = pd.read_csv(path)
    x = data['reviewText'].tolist()
    y = data['sentiment'].tolist()
    return x, y

train_x, train_y = train['text'].tolist(), train['label'].tolist()
test_x, test_y = test['text'].tolist(), test['label'].tolist()

print('training size:', len(train_x))
print('test size:', len(test_x))

# =============================================================================
# plot wordcloud
# =============================================================================
train_pos = train[train['label'] == 'positive']
train_pos = train_pos['text']
train_neg = train[ train['label'] == 'negative']
train_neg = train_neg['text']

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Positive words")
wordcloud_draw(train_pos,'white')
print("Negative words")
wordcloud_draw(train_neg)


# =============================================================================
# Preprocessing
# =============================================================================
lemmatizer = nltk.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
transtbl = str.maketrans(string.punctuation, ' ' * len(string.punctuation)) # remove blamk space

def preprocessing(line: str) -> str:
    line = line.replace('<br />', '').translate(transtbl)
    
    tokens = [lemmatizer.lemmatize(t.lower(),'v')  # What to put in the list
              for t in nltk.word_tokenize(line)    # Where 
              if t.lower() not in stopwords]       # If
    
    return ' '.join(tokens)


#test_str = "I bought several books yesterday<br /> and I really love them!"
#preprocessing(test_str)

# Yet a more modern way to write code
train_x = list(map(preprocessing, train_x))
test_x = list(map(preprocessing, test_x))


my_all_words = [w for line in train_x for w in line.split()]
voca = nltk.FreqDist(my_all_words)

print(voca)

voca.most_common(10)

##出现频率最高的10000个次
topwords = [fpair[0] for fpair in list(voca.most_common(22000))]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn import metrics

# =============================================================================
# Tf-idf term weight
# =============================================================================
# 对权值进行调整

tf_vec = CountVectorizer(vocabulary=topwords)
train_features = tf_vec.fit_transform(train_x)
train_features.shape

# Extract features from test set
test_features = tf_vec.transform(test_x)
test_features.shape
# =============================================================================
# Modeling
# =============================================================================
from sklearn.naive_bayes import BernoulliNB
NB_model = BernoulliNB()

# Train Model
import time

start = time.time()
NB_model.fit(train_features, train_y)
end = time.time()

print("Bernoulli NB model trained in %f seconds" % (end-start))

# Predict
pred = NB_model.predict(test_features)
print(pred)
pred = np.array(list(map(convert_01, pred)))
# Metrics
# metrics.accuracy_score(y_true, y_pred)
from sklearn import metrics
accuracy = metrics.accuracy_score(pred,test_y)
print(accuracy)

# Use keyword arguments to set arguments explicitly
print(metrics.classification_report(y_true=test_y, y_pred=pred))


# =============================================================================
# ## ROC curve
# =============================================================================
# Metrics
# metrics.accuracy_score(y_true, y_pred)
from sklearn import metrics
accuracy = metrics.accuracy_score(pred,test_y)
print(accuracy)

#The list of lists
# Compute ROC curve and ROC area for each class
from sklearn.metrics import roc_curve, auc
n_classes = 2
y_pred = NB_model.predict_proba(test_features)
y_pred[:,0]
y_pred_list = y_pred[:,1]

fpr, tpr, threshold = metrics.roc_curve(test_y, y_pred_list)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)'% roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],color='navy', lw=lw, linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




# =============================================================================
# Predict new sentence
# =============================================================================

# Predict a new sentence
# vectorizer needs to be pre-fitted
# At the end of the project, the function signature should be something like:
# predict_new(sentent: str, vec, model) -> str

def predict_new(sentence: str):
    sentence = preprocessing(sentence)
    features = tf_vec.transform([sentence])
    pred = mnb_model.predict(features)
    return pred[0]


predict_new("Not Good")



s5 = "Easy on your skin and removes make up easily.: I first found these at Trader Joes and picked them up on a whim. I was pleasantly surprised at how well they removed my makeup and left my skin feeling clean and cared for and have often followed with moisturizer after a long day at work so I could at least get my makeup off before I went to bed. Of course for days that I have more time and energy I can follow with my normal cleansing routine.Ive tried a few other brands but have always returned to Comodynes as the other brands would start to irritate my skin even in a weeks time and Ive never experienced any irritation with Comodynes.I ordered the 3-pack this time but will order the 6-pack next time for the value and would definitely buy it through the Subscribe & Save program if it were offered."
predict_new(s5)

s4 = "Probably Lentz's most engaging piece: Missa Umbrarum the title piece uses an interesting process: A number of 30-second electronic delays are used to slowly build up to the final destination for each phrase of the Mass. The nonvocal music comes entirely from crystal glasses (as per a glass harmonica) played by the vocalists. On the first repetition of a phrase the lowest notes of the section are supplied and then the singers drink from the glasses before adding the next layer. In the end each phrase is built from half-a-dozen or more layers turning the handful of vocalists (and glasses) into a full choir. Despite the technique the final result does not sound dense and muddy; an ethereal nature is retained throughout."
predict_new(s4)

s3 = "A waste of money: I had to listen to this by myself. I wouldn't want to subject this music to others. It's not that I'm new to noise/fuzz pop. I own and enjoy many CDs by Sonic Youth My Bloody Valentine Dinosaur Jr. but this band just lacks the ability to put together listenable tunes. I had to sell this sorry collection of tunes after 3 or 4 listens. I gave it a chance but each time I listened to it I lost even more interest."
predict_new(s3)

s2 = "Good Instructional Video: This is actually a really good DVD. It teaches you how to do the moves in a fun way. Highly recommended."
predict_new(s2)

s1 = "There worst yet: i'm a pretty big fan of limp bizkit been listening to them for a few years was very disapointed with this cd. wouldnt reccomened it. try on of there other cds instead"
predict_new(s1)







