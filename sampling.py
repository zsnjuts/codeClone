# -*- coding:utf-8 -*-
"""从数据集中随机采样
"""
import os
import random
import csv
from collections import namedtuple


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
        type_A = random.randint(0, len(type_dirs)-1)  # 类别A：目录A
        txtsA = os.listdir('dataset/train/'+type_dirs[type_A])
        type_B = random.randint(0, len(type_dirs)-1)  # 类别B：目录B
        txtsB = os.listdir('dataset/train/'+type_dirs[type_B])
        while type_A == type_B:
            type_B = random.randint(0, len(type_dirs)-1)
        a = random.randint(0, len(txtsA)-1)  # 从目录A随机选一个样本a
        b = random.randint(0, len(txtsB)-1)  # 从目录B随机选一个样本b
        neg.append(('dataset/train/'+type_dirs[type_A]+'/'+txtsA[a], 'dataset/train/'+type_dirs[type_B]+'/'+txtsB[b]))

    return pos, neg

headings = []
idpairs = []


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
        samples.append(('dataset/test/'+id1, 'dataset/test/'+id2))
    return samples


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
