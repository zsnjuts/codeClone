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
        print(node.args.exprs[0].value+str(len(node.args.children())-1))
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
        print(node.args.exprs[0].value + str(len(node.args.children()) - 1))
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


def feature_extract(filename):
    """提取给定文件的特征向量"""
    ast = parse_file(filename, use_cpp=False)
    return extract_io(ast)


if __name__ == '__main__':
    print(feature_extract('test.txt'))
