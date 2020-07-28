import sys
import os
import scipy.sparse
import sklearn.naive_bayes as NB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import text

########### Reading the data and the labels ###########

UCI_train_data = []
UCI_train_labels = []




########### UCI: Amazon, IMDB, Yelp Reviews ###########
with open("../Data/sentiment labelled sentences/amazon_cells_labelled.txt", 'r') as f:
    content = f.readlines()
    content = [x.strip() for x in content]

for review in content:
    UCI_train_data.append(review.split("\t")[0])
    UCI_train_labels.append(review.split("\t")[1])

with open("../Data/sentiment labelled sentences/imdb_labelled.txt", 'r') as f:
    content = f.readlines()
    content = [x.strip() for x in content]

for review in content:
    UCI_train_data.append(review.split("\t")[0])
    UCI_train_labels.append(review.split("\t")[1])

with open("../Data/sentiment labelled sentences/yelp_labelled.txt", 'r') as f:
    content = f.readlines()
    content = [x.strip() for x in content]

for review in content:
    UCI_train_data.append(review.split("\t")[0])
    UCI_train_labels.append(review.split("\t")[1])


print(UCI_train_data[1])
############ Creating feature vectors ###########
vectorizer = text.TfidfVectorizer(min_df=2,
                             max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=True,
                             # max_features = 10000,
                             ngram_range = (1, 2)
                                  )
                             #stop_words = text.ENGLISH_STOP_WORDS)

UCI_train_vectors = vectorizer.fit_transform(UCI_train_data)
print(UCI_train_vectors[1])


print("UCI Features Size: ", len(UCI_train_vectors.toarray()[0]))
print("\n")


######******************* Calssification Models *******************######

###################### Logistic Regression ####################
logistic_reg = LR()

scores = cross_val_score(logistic_reg, UCI_train_vectors, UCI_train_labels, cv=10, scoring='f1_macro')
#scores = cross_val_score(log_reg, train_vectors, train_labels, cv=5)

print("Logistic Regression F1 Score on UCI dataset: ", scores.mean())
print("\n")

###################### Multilayer Perceptron ######################
ML_perceptron = MLPClassifier()
UCI_scores = cross_val_score(ML_perceptron, UCI_train_vectors, UCI_train_labels, cv=10, scoring='f1_macro')

print("ML Perceptron F1 Score on UCI dataset: ", UCI_scores.mean())
print("\n")

###################### SVM, kernel=linear ######################
classifier_liblinear = svm.LinearSVC()
UCI_scores = cross_val_score(classifier_liblinear, UCI_train_vectors, UCI_train_labels, cv=10, scoring='f1_macro')

print("SVM F1 Score on UCI dataset: ", UCI_scores.mean())
print("\n")

##################### Naive Bayes ######################
bnb = NB.MultinomialNB()
UCI_scores = cross_val_score(bnb, UCI_train_vectors, UCI_train_labels, cv=10, scoring='f1_macro')

print("Naive Bayes F1 Score on UCI dataset: ", UCI_scores.mean())
print("\n")


# ###################### Voting Classifier ######################
# voting_clf = VotingClassifier(estimators=[('nb', bnb), ('lg1', logistic_reg), ('svc', classifier_liblinear), ('mlp', ML_perceptron)],
#                        voting='hard', weights=[1,1,1,1])
# # voting_clf = VotingClassifier(estimators=[('nb', bnb), ('lg1', logistic_reg), ('svc', classifier_liblinear)],
# #                       voting='hard', weights=[1,1,1])
# UCI_scores = cross_val_score(voting_clf, UCI_train_vectors, UCI_train_labels, cv=10, scoring='f1_macro')
#
# print("Voting Classifier F1 Score on UCI dataset: ", UCI_scores.mean())