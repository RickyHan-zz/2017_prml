import tensor_nn as tn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def run_svm(ax, train_x, test_x, train_y, test_y, k='rbf', penalty=1.0, g=0.2, d=9):
    clf = svm.SVC(kernel=k, gamma=g, degree=d, C=penalty).fit(train_x, train_y)
    score = cross_val_score(clf, test_x, test_y, cv=5)
    print(score)
    if ax != None:
        ax.set_ylabel("Accuracy")
        ax.plot(score)
    return np.mean(score) * 100.


def run_dt(train_x, test_x, train_y, test_y):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(train_x, train_y)
    scores = cross_val_score(clf, test_x, test_y, cv=5)
    return np.mean(scores) * 100.


def run_gnb(train_x, test_x, train_y, test_y):
    gnb = GaussianNB()
    gnb.fit(train_x, train_y)
    scores = cross_val_score(gnb, test_x, test_y, cv=5)
    return np.mean(scores) * 100.

def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))
    full_data = tn.load_data()
    small_data = tn.load_data(use_small_set=True)
    splitted_full_data = tn.split_data(*full_data)
    splitted_small_data = tn.split_data(*small_data)
    full_accuracy = []
    # 2layer MLP
    full_accuracy.append(tn.run_nn(None, *tn.split_data(*full_data, hot_encoding = True)))
    #SVM
    full_accuracy.append(run_svm(None, *splitted_full_data, k='linear'))
    full_accuracy.append(run_svm(None, *splitted_full_data, k='rbf'))
    full_accuracy.append(run_svm(None, *splitted_full_data, k='poly'))
    #DT
    full_accuracy.append(run_dt(*splitted_full_data))
    #GNB
    full_accuracy.append(run_gnb(*splitted_full_data))
    print(full_accuracy)

    #2classes
    small_accuracy = []
    small_accuracy.append(tn.run_nn(None, *tn.split_data(*small_data, hot_encoding = True)))
    # SVM
    small_accuracy.append(run_svm(None, *splitted_small_data, k='linear'))
    small_accuracy.append(run_svm(None, *splitted_small_data, k='rbf'))
    small_accuracy.append(run_svm(None, *splitted_small_data, k='poly'))
    # DT
    small_accuracy.append(run_dt(*splitted_small_data))
    # GNB
    small_accuracy.append(run_gnb(*splitted_small_data))
    print(small_accuracy)

    # Result
    kind = ('MLP', 'SVM(Linear)', 'SVM(RBF)', 'SVM(POLY)', 'Decision Tree', 'Naive Bayes')
    pos = np.arange(len(kind))
    width = 0.5
    #Plots for 9 classes
    ax1.set_title('Overall Performance (9 classes)')
    ax1.set_ylabel('Accuracy')
    ax1.bar(pos, full_accuracy, align='center', width=width)
    ax1.set_xticks(pos + width / 2)
    ax1.set_xticklabels(kind)
    for i in range(len(full_accuracy)):
        ax1.annotate('{:.2f}'.format(full_accuracy[i]), xy=(i, full_accuracy[i]), textcoords='data', color='blue')

    #plots for 2classes
    ax2.set_title('Overall Performance (2 classes)')
    ax2.set_ylabel('Accuracy')
    ax2.bar(pos, small_accuracy, align='center', width=width)
    ax2.set_xticks(pos + width / 2)
    ax2.set_xticklabels(kind)
    for i in range(len(small_accuracy)):
        ax2.annotate('{:.2f}'.format(small_accuracy[i]), xy=(i, small_accuracy[i]), textcoords='data', color='blue')

    plt.savefig('plots.png')
    plt.show()


if __name__ == '__main__':
    main()