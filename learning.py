# -*- coding:utf-8 -*-
"""使用sklearn学习样本，输出test结果"""
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sampling import *
import pickle
import os


def learning(samples, labels, testin):
    """learn"""
    modelpath = 'dataset/model'
    if os.path.exists(modelpath):
        print('hit cache:dataset/model')
        with open(modelpath, 'rb') as f:
            clf = pickle.load(f)
    else:
        print('generating model...')
        clf = RandomForestClassifier()
        clf.fit(samples, labels)
        with open(modelpath, 'wb') as f:
            pickle.dump(clf, f)
    print(cross_val_score(RandomForestClassifier(), samples, labels))
    return clf.predict(testin)


if __name__ == '__main__':
    trainsamples, trainlabels = get_training_samples(5000, 50000)
    # testsamples = []
    testsamples = get_test_samples()
    # learning(trainsamples, trainlabels, testsamples)
    testlabels = learning(trainsamples, trainlabels, testsamples)
    set_test_result(testlabels)
