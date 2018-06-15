# -*- coding:utf-8 -*-
"""对比文档相似度"""
import re
import pickle
import time
from collections import namedtuple

from pycparser.c_ast import *
import csv


# 阈值为19：presicion=0.19196003805899145,recall=0.807；f1=0.31014604150653347
# 阈值为20：presicion=0.22856447392442036,recall=0.7512；f1=0.3504875659030467
# 阈值为21：presicion=0.27555519923033756,recall=0.6874；f1=0.393406970754879
# 阈值为22：presicion=0.3347388906908855,recall=0.6192；f1=0.434556811004281；time cost:39.62852692604065
# 阈值为23：presicion=0.4045936395759717,recall=0.5496；f1=0.466078697421981；time cost:36.20327115058899
# 阈值为24：presicion=0.4857142857142857,recall=0.476；f1=0.4808080808080808；time cost:37.77420210838318
# 阈值为25：presicion=0.5750354609929078,recall=0.4054；f1=0.4755425219941349；time cost:37.770132064819336；kaggle评定值为0.46816
def predict_doc_similarity(file1, file2):
    with open(file1) as f:
        text1 = f.read()
    with open(file2) as f:
        text2 = f.read()
    wordset1 = set(re.split(' |\n|\r|\t|,|;|\(|\)|\{|\}', text1))
    wordset2 = set(re.split(' |\n|\r|\t|,|;|\(|\)|\{|\}', text2))
    return len(wordset1 & wordset2) >= 25


def get_lexset(node):
    lexset = set([])
    if node.attr_names:
        for n in node.attr_names:
            lexset.add(str(getattr(node, n)))
    for _, child in node.children():
        lexset |= get_lexset(child)
    return lexset


# 阈值为19：presicion=0.202627345844504,recall=0.7559511902380476；f1=0.31959068036703453；time cost:595.5326461791992
# 阈值为20：presicion=0.24727011494252873,recall=0.6885377075415083；f1=0.36386701199852006；time cost:161.42496299743652
# 阈值为21：presicion=0.3047562543261149,recall=0.6165233046609322；f1=0.4078877713075702；time cost:166.43155431747437
# 阈值为22：presicion=0.3867884750527055,recall=0.5505101020204041；f1=0.45435033845137857；time cost:149.31481790542603
# 阈值为23：presicion=0.4789334418888663,recall=0.47069413882776556；f1=0.47477804681194513；time cost:166.28159523010254
# 阈值为24：presicion=0.5788059701492537,recall=0.38787757551510305；f1=0.4644867648820218；time cost:154.35775423049927
def predict_ast_similarity(file1, file2):
    path1 = file1.split('/')
    if path1[1] == 'train':
        astpath1 = path1[0]+'/'+path1[1]+'_ast/'+path1[2]+'/'+path1[3]+'.ast'
    else:
        astpath1 = path1[0] + '/' + path1[1] + '_ast/' + path1[2] + '.ast'
    with open(astpath1, 'rb') as f:
        ast1 = pickle.load(f)
    path2 = file2.split('/')
    if path2[1] == 'train':
        astpath2 = path2[0]+'/'+path2[1]+'_ast/'+path2[2]+'/'+path2[3]+'.ast'
    else:
        astpath2 = path2[0] + '/' + path2[1] + '_ast/' + path2[2] + '.ast'
    with open(astpath2, 'rb') as f:
        ast2 = pickle.load(f)
    return len(get_lexset(ast1) & get_lexset(ast2)) >= 24


def train_evaluate(predict):
    with open('dataset/train_filepairs_p5n50', 'rb') as f:
        print('hit cache')
        pos, neg = pickle.load(f)
    begin = time.time()
    fn, tp = 0, 0
    for file1, file2 in pos:
        try:
            if not predict(file1, file2):
                fn += 1
        except EOFError:
            tp -= 1
    tp += len(pos) - fn

    fp, tn = 0, 0
    for file1, file2 in neg:
        try:
            if predict(file1, file2):
                fp += 1
        except EOFError:
            tn -= 1
    tn += len(neg) - fp

    presicion = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = 2*presicion*recall/(presicion+recall)
    print('tp=%d, fn=%d, fp=%d, tn=%d' % (tp, fn, fp, tn))
    print('presicion='+str(presicion)+',recall='+str(recall))
    print('f1='+str(f1))
    print('time cost:'+str(time.time()-begin))


def set_test_result(predict):
    fcsv = csv.reader(open('dataset/test_result.csv'))
    headings = next(fcsv)
    Row = namedtuple('Row', headings)
    rows = []
    for i, r in enumerate(fcsv):
        id1_id2 = Row(*r).id1_id2
        id1, id2 = id1_id2.split('_')
        file1, file2 = 'dataset/test/' + id1 + '.txt', 'dataset/test/' + id2 + '.txt'
        print(str(i+1) + '/200000:' + file1 + ' - ' + file2)
        rows.append((id1_id2, int(predict(file1, file2))))
    fcsv = csv.writer(open('dataset/test_result.csv', 'w', newline=''))
    fcsv.writerow(headings)
    fcsv.writerows(rows)


if __name__ == '__main__':
    set_test_result(predict_ast_similarity)
    # train_evaluate(predict_ast_similarity)
    # print(predict_ast_similarity('dataset/train/bb7a/7c7056fea3574b61.txt', 'dataset/train/bb7a/2efc5ff55e4d459d.txt'))
