# -*- coding=utf-8 -*-
"""提取特征向量"""
from pycparser import parse_file
from pycparser.c_ast import *
import os
import pickle


def is_cin(node):
    """递归检查此结点是否为cin输入结点"""
    if not isinstance(node, BinaryOp):
        return False
    elif isinstance(node.left, ID) and node.left.name == 'cin':
        return True
    elif isinstance(node.left, BinaryOp) and node.left.op == '>>':
        return is_cin(node.left)
    else:
        return False


def is_cout(node):
    """递归检查此结点是否为cout输入结点"""
    if not isinstance(node, BinaryOp):
        return False
    elif isinstance(node.left, ID) and node.left.name == 'cout':
        return True
    elif isinstance(node.left, BinaryOp) and node.left.op == '<<':
        return is_cout(node.left)
    else:
        return False


def extract_stdin(node):
    """从AST中提取输入变量数:直接输入，循环输入，条件输入"""
    directin, loopin, condin = 0, 0, 0
    # scanf数量
    if isinstance(node, FuncCall) and node.name.name == 'scanf':
        directin += len(node.args.children())-1
    # cin数量
    if is_cin(node):
        directin += 1
    for _, child in node.children():
        din, lin, cin = extract_stdin(child)
        if isinstance(child, For) or isinstance(child, While) or isinstance(child, DoWhile):
            loopin += din+lin+cin
            condin += cin
        elif isinstance(child, If) or isinstance(child, Switch):
            condin += din+lin+cin
            loopin += lin
        else:
            directin += din
            loopin += lin
            condin += cin
    return directin, loopin, condin


def extract_stdout(node):
    """从AST中提取输出变量数:直接输出，循环输出，条件输出"""
    directout, loopout, condout = 0, 0, 0
    # printf数量
    if isinstance(node, FuncCall) and node.name.name == 'printf':
        directout += len(node.args.children()) - 1
    # cout数量
    if is_cout(node):
        directout += 1
    for _, child in node.children():
        dout, lout, cout = extract_stdout(child)
        if isinstance(child, For) or isinstance(child, While) or isinstance(child, DoWhile):
            loopout += dout + lout + cout
            condout += cout
        elif isinstance(child, If) or isinstance(child, Switch):
            condout += dout + lout + cout
            loopout += lout
        else:
            directout += dout
            loopout += lout
            condout += cout
    return directout, loopout, condout


def extract_io(ast):
    """提取输入输出相关特征"""
    directin, loopin, condin = extract_stdin(ast)
    directout, loopout, condout = extract_stdout(ast)
    return [directin, loopin, condin, directout, loopout, condout]


def extract_text(node):
    """提取代码中所有字符串(包括无参数printf)和单个字符"""
    strsum, charsum = 0, 0
    if node.attr_names.__contains__('type'):
        if node.type == 'string' and not ('%' in node.value):
            string = node.value.replace('"', '')
            # print('string:'+string)
            strsum += sum([ord(c) for c in string])
        elif node.type == 'char':
            string = node.value.replace("'", "")
            if len(string) > 1:
                if string[1] == 'n':
                    string = '\n'
                elif string[1] == 'r':
                    string = '\r'
                elif string[1] == 't':
                    string = '\t'
                elif string[1] == '0':
                    string = '\0'
            # print('char:'+string)
            charsum += ord(string)
    for _, child in node.children():
        sh, ch = extract_text(child)
        strsum += sh
        charsum += ch
    return [strsum, charsum]


def extract_iter(node):
    """最大循环深度"""
    depth = 0
    if isinstance(node, For) or isinstance(node, While) or isinstance(node, DoWhile):
        depth += 1
    if len(node.children()):
        depth += max([extract_iter(child) for _, child in node.children()])
    return depth


def extract_calc(node):
    """各种运算符出现次数[+ - * / % ++]"""
    typecount = [0]*5
    if isinstance(node, BinaryOp):
        if node.op == '+':
            typecount[0] += 1
        elif node.op == '-':
            typecount[1] += 1
        elif node.op == '*':
            typecount[2] += 1
        elif node.op == '/':
            typecount[3] += 1
        elif node.op == '%':
            typecount[4] += 1
    for _, child in node.children():
        tpcnt = extract_calc(child)
        typecount = [typecount[i]+tpcnt[i] for i in range(0, len(typecount))]
    return typecount


def extract_arr(node):
    """统计申请数组的大小"""
    size = 0
    if isinstance(node, ArrayDecl):
        if isinstance(node.dim, Constant):
            try:
                size = int(node.dim.value)
            except ValueError as err:
                size = ord(node.dim.value.replace("'", ""))
        # else:
        #     print('dim is not constant')
    elif isinstance(node, Struct):  # 避免Struct定义中的数组影响结果
        return 0

    for _, child in node.children():
        size = max(size, extract_arr(child))
    return size


def extract_constant(node):
    """统计代码中所有常量之和"""
    sm = 0
    if isinstance(node, Constant):
        try:
            sm += int(node.value)
        except ValueError as err:
            pass

    for _, child in node.children():
        sm += extract_constant(child)
    return sm


def extract_type(node):
    """分别统计循环和分支的数目"""
    typecount = [0]*2
    if isinstance(node, Break) or isinstance(node, Continue) \
       or isinstance(node, Case) or isinstance(node, Default) \
       or isinstance(node, If):
        typecount[0] += 1
    elif isinstance(node, DoWhile) or isinstance(node, For) or isinstance(node, While):
        typecount[1] += 1

    for _, child in node.children():
        tpcnt = extract_type(child)
        typecount = [typecount[i]+tpcnt[i] for i in range(0, len(typecount))]
    return typecount


def extract_basic(node):
    basiccount = [0]*4
    if isinstance(node, IdentifierType):
        for name in node.names:
            if name == 'int':
                basiccount[0] += 1
            elif name == 'float':
                basiccount[1] += 1
            elif name == 'char':
                basiccount[2] += 1
            elif name == 'double':
                basiccount[3] += 1

    for _, child in node.children():
        bsccnt = extract_basic(child)
        basiccount = [basiccount[i]+bsccnt[i] for i in range(0, len(basiccount))]
    return basiccount


def get_func(node):
    """统计代码中所有被调用的函数名"""
    funcs = set()
    if isinstance(node, FuncCall) and node.name.name != 'scanf' and node.name.name != 'printf':
        funcs.add(node.name.name)
    for _, child in node.children():
        funcs |= get_func(child)
    return funcs


# def get_basic(node):
#     """统计代码中出现的所有基本类型"""
#     basics = set([])
#     if isinstance(node, IdentifierType):
#         basics |= set(node.names)
#     for _, child in node.children():
#         basics |= get_basic(child)
#     return basics


# p5n50，相减，带上constant交叉验证[0.94783083 0.94750341 0.9481526], kaggle提交后有提升(0.53453->0.54748)
# p5n50，相减，带上constant和循环分支type统计交叉验证[0.94870396 0.94875853 0.94924412]
# p5n50，相减，带上constant和循环分支type统计和函数交集func交叉验证[0.95099591 0.95094134 0.95366479]
# p5n50，相减，带上constant和循环分支type统计和函数交集func和基本类型数basic交叉验证[0.95126876 0.95214188 0.95284615]
# p40n400，相减，带上constant和循环分支type统计和函数交集func和基本类型数basic交叉验证[0.96535211 0.96454689 0.96501736]，
#   kaggle提交分数有提升(0.54968->0.60742)
# p5n50，相减，带上constant和循环分支type统计和函数交集func加上basic交集统计交叉验证有下降
# p5n50，相减，带上constant和循环分支type统计和函数交集func,100棵树交叉验证[0.9544884  0.95427012 0.95590242]
# p10n100，相减，带上constant交叉验证[0.95181446 0.951322   0.95063851]，kaggle提交后有提升[0.54748->0.54859]
# p40n400, 相减，带上constant交叉验证[0.95647013 0.95670178 0.95677652]，kaggle提交后有提升[0.54859->0.54968]
# 再带上type交叉验证[0.9427558  0.94450205 0.94487802],kaggle提交之后只有0.54019
def feature_extract(filename):
    """提取给定文件的特征向量"""
    # print(filename)
    # ast = parse_file(filename, use_cpp=False)
    paths = filename.split('/')
    name = ''
    if paths[1] == 'test':
        name = 'dataset/test_ast/'+paths[-1]+'.ast'
    elif paths[1] == 'train':
        name = 'dataset/train_ast/'+paths[-2]+'/'+paths[-1]+'.ast'
    if os.path.exists(name):
        with open(name, 'rb') as f:
            ast = pickle.load(f)
        return extract_io(ast) + extract_text(ast) + [extract_iter(ast)]\
            + extract_calc(ast) + [extract_constant(ast)] + extract_type(ast) + extract_basic(ast), get_func(ast)
    else:
        raise Exception('ast not cached')


def get_sample_feature(file1, file2):
    """生成文件对样本特征"""
    f1, fc1 = feature_extract(file1)
    f2, fc2 = feature_extract(file2)
    return [abs(f1[i]-f2[i]) for i in range(len(f1))] + [len(fc1 & fc2)]


if __name__ == '__main__':
    print(get_sample_feature('dataset/train/a98e/0bf3f6043f564c86.txt', 'dataset/train/a98e/2c715313d06340ea.txt'))
    # import profile
    # profile.run(
    #     "get_sample_feature('dataset/train/0ae1/0ade50fef00347e7.txt', 'dataset/train/1a4d/0bd9a05fe6914906.txt')")
