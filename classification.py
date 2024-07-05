import os
from numpy import concatenate
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from utils import *  # here the code communicates with utils.py

# csv sources
files = ["./data/class1/vecs.csv", "./data/class2/vecs.csv"]

# fetch data in csv and structure in a class:vector dict
cl_dict = get_data(files)
t1 = list(cl_dict.keys())[0]
t2 = list(cl_dict.keys())[1]

# hard code the training set to 80% and test set to 20%
train1_size = int(0.8 * len(cl_dict[t1]))  
train2_size = int(0.8 * len(cl_dict[t2]))

print(f'Train sizes - Topic 1: {train1_size}, Topic 2: {train2_size}')

# build numpy arrays and lists of docs
t1_train, t1_test, t1_train_docs, t1_test_docs = make_arrays(cl_dict[t1], train1_size)
t2_train, t2_test, t2_train_docs, t2_test_docs = make_arrays(cl_dict[t2], train2_size)
train_docs = t1_train_docs + t2_train_docs

test1_size = len(t1_test)
test2_size = len(t2_test)

print(
    'Topic 1: Train size: {} | Test size: {}\n'.format(train1_size, test1_size) +
    'Topic 2: Train size: {} | Test size: {}\n'.format(train2_size, test2_size)
)

# prepare train/test sets
x_train = concatenate([t1_train, t2_train])
x_test = concatenate([t1_test, t2_test])
y_train = make_labels(train1_size, train2_size)
y_test = make_labels(test1_size, test2_size)

# define parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf'],
    'degree': [2, 3, 4]  # only relevant for 'poly' kernel
}

# setup SVM with GridSearchCV
print('SVC output:')
svc = SVC(verbose=True)
clf = GridSearchCV(svc, param_grid, cv=5, verbose=3)
clf.fit(x_train, y_train)

# best model and score
best_model = clf.best_estimator_
best_score = clf.best_score_

print('\n')
print('Best SVC Model:')
print(best_model)
print('Best cross-validation score: {}'.format(best_score))

# evaluate the best model on the test set
score = best_model.score(x_test, y_test)
y_pred = best_model.predict(x_test)

print('Score on test set: {}\n'.format(score))

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix:')
print(conf_matrix)
