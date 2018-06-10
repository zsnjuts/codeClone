# -*- coding=utf-8 -*-
"""提取特征向量"""
from pycparser import parse_file
from pycparser.c_ast import *


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
    """各种运算符出现次数[+ - * / %]"""
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


def feature_extract(filename):
    """提取给定文件的特征向量"""
    # print(filename)
    ast = parse_file(filename, use_cpp=False)
    return extract_io(ast)+extract_text(ast)+[extract_iter(ast)]+extract_calc(ast)


if __name__ == '__main__':
    print(feature_extract('dataset/train/5593/a0a642ee43d44b34.txt'))
