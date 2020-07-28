
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import re, string, collections, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

path = "../Data/sentiment labelled sentences"
filepath_dict = {'yelp':   '../Data/sentiment labelled sentences/yelp_labelled.txt',
                 'amazon': '../Data/sentiment labelled sentences/amazon_cells_labelled.txt',
                 'imdb':   '../Data/sentiment labelled sentences/imdb_labelled.txt'}
df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df_list.append(df)
# df_list
df = pd.concat(df_list)
sentences = ['Rashmi likes ice cream', 'Rashmi hates chocolate.']
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
print(vectorizer.vocabulary_)
print(vectorizer.transform(sentences).toarray())

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


X_train, X_test, y_train, y_test = train_test_split(df['sentence'].values,
                 df['label'].values,
                 test_size=0.1)

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

vect = CountVectorizer(tokenizer=tokenize)
tf_train = vect.fit_transform(X_train)
tf_test = vect.transform(X_test)
# print(tf_train)
# print(tf_train[0])
vocab = vect.get_feature_names()
print(len(vocab))
w0 = set([o for o in X_train[0].split(' ')])
vect.vocabulary_['unless']
svd = TruncatedSVD()
reduced_tf_train = svd.fit_transform(tf_train)
p = tf_train[y_train==1].sum(0) + 1
q = tf_train[y_train==0].sum(0) + 1
r = np.log((p/p.sum())/(q/q.sum()))
b = np.log(len(p)/len(q))
pre_preds = tf_test @ r.T + b
preds = pre_preds.T > 0
acc = (preds==y_test).mean()
print(f'Accuracy: {acc}')

# model = LogisticRegression(C=0.2, dual=True)
# model.fit(tf_train, y_train)
# preds = model.predict(tf_test)
# acc = (preds==y_test).mean()
# print(f'Accuracy: {acc}')

from sklearn.metrics import classification_report
print(y_test)
print("----")
preds = np.where(preds==True, 1, 0)
preds = preds.T
print("Precision, Recall and F1-Score:\n\n",classification_report(y_test, preds))