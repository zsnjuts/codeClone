# -*- coding:utf-8 -*-
"""使用sklearn学习样本，输出test结果"""
from sklearn import svm
from sampling import *
from feature import *


def learning(samples, labels, testin):
    """learn"""
    clf = svm.SVC()
    clf.fit(samples, labels)
    return clf.predict(testin)


if __name__ == '__main__':
    posnumber, negnumber = 50, 50
    trainsamples, trainlabels = get_training_samples(posnumber, negnumber)
    testsamples = get_test_samples()
    testlabels = learning(trainsamples, trainlabels, testsamples)
    set_test_result(testlabels)
