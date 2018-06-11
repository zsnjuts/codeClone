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
from pycparser import parse_file


def cache(path):
    """为函数运行结果生成cache"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            if os.path.exists(path):
                print('hit cache:'+path)
                with open(path, 'rb') as f:
                    return pickle.load(f)
            else:
                train_dataset = func(*args, **kw)
                with open(path, 'wb') as f:
                    pickle.dump(train_dataset, f)
                return train_dataset
        return wrapper
    return decorator


@cache('dataset/train_filepairs_p20n20')
def get_training_filepairs(posnumber, negnumber):
    """生成训练集的文件名对"""
    type_dirs = os.listdir('dataset/train')

    print('selecting positive file pairs...')
    pos = []  # 正例
    for i in range(0, posnumber):
        type_no = random.randint(0, len(type_dirs)-1)  # 类别目录索引
        txts = os.listdir('dataset/train/'+type_dirs[type_no])
        a = random.randint(0, len(txts)-1)  # 样本a
        b = random.randint(0, len(txts)-1)  # 样本b
        while a == b:
            b = random.randint(0, len(txts) - 1)
        pos.append(('dataset/train/'+type_dirs[type_no]+'/'+txts[a], 'dataset/train/'+type_dirs[type_no]+'/'+txts[b]))

    print('selecting negative file pairs...')
    neg = []  # 负例
    for i in range(0, negnumber):
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

    return pos, neg


@cache('dataset/train_dataset_p20n20')
def get_training_samples(posnumber, negnumber):
    """生成训练样本"""
    pos, neg = get_training_filepairs(posnumber, negnumber)
    print('building samples...')
    samples = []
    # 生成正例
    for i, (a, b) in enumerate(pos):
        print(str(i+1)+r'/'+str(posnumber+negnumber)+':'+a+' - '+b)
        try:
            fa, fb = feature_extract(a), feature_extract(b)
        except Exception as err:  # 出现错误，则跳过此样本
            print('Error:'+str(err))
            continue
        samples.append(feature_extract(a)+feature_extract(b))
    posnum = len(samples)  # 由于中间有的训练集存在语法错误，所以可能生成的samples与pos不相等
    # 生成负例
    for i, (a, b) in enumerate(neg):
        print(str(posnumber+i+1) + r'/' + str(posnumber+negnumber) + ':' + a + ' - ' + b)
        try:
            fa, fb = feature_extract(a), feature_extract(b)
        except Exception as err:  # 出现错误，则跳过此样本
            print(err)
            continue
        samples.append(feature_extract(a) + feature_extract(b))
    negnum = len(samples) - posnum
    labels = [1]*posnum + [0]*negnum
    print(str(len(samples))+' samples generated')
    return samples, labels


@cache('dataset/test_samples')
def get_test_samples():
    """获取test集中的所有样本"""
    fcsv = csv.reader(open('dataset/test_result.csv'))
    headings = next(fcsv)
    Row = namedtuple('Row', headings)
    samples = []
    for r in fcsv:
        id1_id2 = Row(*r).id1_id2
        id1, id2 = id1_id2.split('_')
        samples.append(('dataset/test/'+id1+'.txt', 'dataset/test/'+id2+'.txt'))
    return [feature_extract(a)+feature_extract(b) for a, b in samples]


def set_test_result(testout):
    """按照原先test集的顺序写入样本预测结果"""
    print('setting test result...')
    fcsv = csv.reader(open('dataset/test_result.csv'))
    headings = next(fcsv)
    Row = namedtuple('Row', headings)
    idpairs = [Row(*r).id1_id2 for r in fcsv]
    rows = []
    for i, id1_id2 in enumerate(idpairs):
        rows.append((id1_id2, testout[i]))
    fcsv = csv.writer(open('dataset/test_result.csv', 'w', newline=''))
    fcsv.writerow(headings)
    fcsv.writerows(rows)

def gen_train_ast():
    """生成训练集的AST"""
    for dr in os.listdir('dataset/train/'):
        print('generating '+dr+'...')
        os.makedirs('dataset/train_ast/'+dr)
        for file in os.listdir('dataset/train/'+dr):
            with open('dataset/train_ast/'+dr+'/'+file+'.ast', 'wb') as f:
                try:
                    pickle.dump(parse_file('dataset/train/' + dr + '/' + file), f)
                except Exception as err:
                    print(err)
                    continue


def gen_test_ast():
    """生成测试集的AST"""
    for file in os.listdir('dataset/test/'):
        with open('dataset/test_ast/'+file+'.ast', 'wb') as f:
            print('generating '+file)
            try:
                pickle.dump(parse_file('dataset/test/' + file), f)
            except Exception as err:
                print(err)
                continue

if __name__ == '__main__':
    with open('dataset/train_dataset_p5n5', 'rb') as f:
        samples5, labels5 = pickle.load(f)
    with open('dataset/train_dataset_p15n15', 'rb') as f:
        samples15, labels15 = pickle.load(f)
    samples = samples5+samples15
    labels = labels5+labels15
    with open('dataset/train_dataset_p20n20', 'wb') as f:
        pickle.dump((samples, labels), f)
