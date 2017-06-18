# coding: utf-8
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.decomposition import FactorAnalysis
from sklearn.naive_bayes import GaussianNB
import graphviz as gv
from sklearn import preprocessing
import matplotlib.pyplot as plt



"Functions for preprocessing data class"
def convert_to_class(code_nm):
   init_class = 0
   if code_nm == u'가로보':
       return init_class + 1
   elif code_nm == u'교각':
       return init_class + 2
   elif code_nm == u'교량받침':
       return init_class + 3
   elif code_nm == u'교명포장':
       return init_class + 4
   elif code_nm == u'난간연석':
       return init_class + 5
   elif code_nm == u'바닥판':
       return init_class + 6
   elif code_nm == u'배수시설':
       return init_class + 7
   elif code_nm == u'신축이음':
       return init_class + 8
   elif code_nm == u'주형':
       return init_class + 9
   return init_class


"Indexes are excel file column numbers"
feature_index = [7, 8, 9, 10, 11]
class_index = 14
num_class = 9

"Data Load"
file = pd.read_excel('bridge_data_2.xlsx')

"x = features / y = class"
x = file[file.columns[feature_index]].values.astype(float)
y = np.reshape(np.array(list(map(convert_to_class, file[file.columns[class_index]].values))), [file.shape[0], 1]).ravel()


"For using only one sample per bridge (Unique)"
# If you want to use all data, comment out this part and use the original x, y variables!!
b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
_, idx = np.unique(b, return_index=True)
unique_x = x[idx]
unique_y = y[idx]

"Scale the each features of data"
# unique_x = preprocessing.scale(unique_x)

"For using only 2 classes, which are the most two common problems (Original)"

index56 = []
for i, j in enumerate(y):
    if j == 5 or j == 6:
        index56.append(i)

idx_56 = np.asarray(index56)
unique_x56 = x[idx_56]
unique_y56 = y[idx_56]

"This is for the dataset having only one sample per bridge (Unique)"
# index56 = []
# for i, j in enumerate(unique_y):
#     if j == 5 or j == 6:
#         index56.append(i)
#
# idx_56 = np.asarray(index56)
# unique_x56 = unique_x[idx_56]
# unique_y56 = unique_y[idx_56]



"Split the Data for Cross validation: Unique/ Original"
X_train, X_test, y_train, y_test = train_test_split(unique_x56, unique_y56, test_size=0.2, random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

"Change the parameter C with these values"
C = 1   # C = 0.1, 1.0, 10, 100
"Change the gamma parameter"
gamma = 0.2 # 0.2, 0.7

"SVM for three kernels"
# clf_linear = svm.LinearSVC(C=C).fit(X_train, y_train)
clf_rbf = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(X_train, y_train)
# clf_poly = svm.SVC(kernel='poly', degree=9, C=C).fit(X_train, y_train)

# scores = cross_val_score(clf_linear, X_test, y_test, cv=5)
scores = cross_val_score(clf_rbf, X_test, y_test, cv=5)
# scores = cross_val_score(clf_poly, X_test, y_test, cv=5)
print(scores)


"Decision Tree Classifier"
# clf = DecisionTreeClassifier(random_state=0)
# clf.fit(X_train, y_train)
# scores = cross_val_score(clf, X_test, y_test, cv=5)

"Gaussian Naive Bayes"
# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, y_train)
# scores = cross_val_score(gnb, X_test, y_test, cv=5)
#
# # print("Number of mislabeled points out of a total %d points : %d"
# #       % (X_test.shape[0], (y_test != y_pred).sum()))
# print(scores)
