# -*- coding:utf-8 -*-
"""使用sklearn学习样本，输出test结果"""
from sklearn import svm
from sampling import *
from feature import *


def learning(samplepaths, labels, testinpaths):
    """learn"""
    clf = svm.SVC()
    samples = [feature_extract(path) for path in samplepaths]
    testin = [feature_extract(path) for path in testinpaths]
    clf.fit(samples, labels)
    testout = clf.predict(testin)
    # testout = [0 for i in range(0, len(testin))]
    return testout


if __name__ == '__main__':
    posnumber, negnumber = 50, 50
    pos, neg = get_training_samples(posnumber, negnumber)
    poslabels = [1 for i in range(0, posnumber)]
    neglabels = [0 for i in range(0, negnumber)]
    tin = get_test_samples()
    tout = learning(pos+neg, poslabels+neglabels, tin)
    set_test_result(tout)
