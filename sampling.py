# -*- coding:utf-8 -*-
"""从数据集中随机采样
"""
import os
import random
import csv
from collections import namedtuple
import pickle
from feature import feature_extract
import functools


def cache(path):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    return pickle.load(f)
            else:
                train_dataset = func(*args, **kw)
                with open(path, 'wb') as f:
                    pickle.dump(train_dataset, f)
                return train_dataset
        return wrapper
    return decorator


@cache('dataset/train_dataset')
def get_training_samples(posNumber, negNumber):
    """sampling filenames from train set
    posNumber为正例数量，negNumber为负例数量
    """
    type_dirs = os.listdir('dataset/train')

    pos = []  # 正例
    for i in range(0, posNumber):
        type_no = random.randint(0, len(type_dirs)-1)  # 类别目录索引
        txts = os.listdir('dataset/train/'+type_dirs[type_no])
        a = random.randint(0, len(txts)-1)  # 样本a
        b = random.randint(0, len(txts)-1)  # 样本b
        while a == b:
            b = random.randint(0, len(txts) - 1)
        pos.append(('dataset/train/'+type_dirs[type_no]+'/'+txts[a], 'dataset/train/'+type_dirs[type_no]+'/'+txts[b]))

    neg = []  # 负例
    for i in range(0, negNumber):
        # 随机选一个目录1，在目录1中随机选一个文件
        type1 = random.randint(0, len(type_dirs)-1)
        txts1 = os.listdir('dataset/train/'+type_dirs[type1])
        a = random.randint(0, len(txts1) - 1)
        # 随机选一个目录2，在目录2中随机选一个文件
        type2 = random.randint(0, len(type_dirs)-1)
        while type1 == type2:
            type2 = random.randint(0, len(type_dirs)-1)
        txts2 = os.listdir('dataset/train/' + type_dirs[type2])
        b = random.randint(0, len(txts2)-1)  # 从目录B随机选一个样本b
        neg.append(('dataset/train/'+type_dirs[type1]+'/'+txts1[a], 'dataset/train/'+type_dirs[type2]+'/'+txts2[b]))

    return [feature_extract(a)+feature_extract(b) for a, b in pos+neg], [1]*len(pos)+[0]*len(neg)

headings = []
idpairs = []


@cache('dataset/test_samples')
def get_test_samples():
    """获取test集中的所有样本"""
    global headings, idpairs
    fcsv = csv.reader(open('dataset/test_result.csv'))
    headings = next(fcsv)
    Row = namedtuple('Row', headings)
    samples = []
    for r in fcsv:
        id1_id2 = Row(*r).id1_id2
        id1, id2 = id1_id2.split('_')
        idpairs.append(id1_id2)
        samples.append(('dataset/test/'+id1+'.txt', 'dataset/test/'+id2+'.txt'))
    return [feature_extract(a)+feature_extract(b) for a, b in samples]


def set_test_result(testout):
    """按照原先test集的顺序写入样本预测结果"""
    global headings, idpairs
    rows = []
    for i, id1_id2 in enumerate(idpairs):
        rows.append((id1_id2, testout[i]))
    fcsv = csv.writer(open('dataset/test_result.csv', 'w', newline=''))
    fcsv.writerow(headings)
    fcsv.writerows(rows)


if __name__ == '__main__':
    pos, neg = get_training_samples(5, 5)
    print(pos)
    print(neg)
