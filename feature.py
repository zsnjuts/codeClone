# -*- coding=utf-8 -*-
"""提取特征向量"""
from pycparser import parse_file
from pycparser.c_ast import *


# def cppin_number(node, depth):
#     """递归计算cin输入的变量数"""
#     if isinstance(node.left, BinaryOp):
#         print(node.left.op)
#     if isinstance(node.left, ID) and node.left.name == 'cin':
#         return depth+1
#     elif isinstance(node.left, BinaryOp) and node.left.op == '>>':
#         return cppin_number(node.left, depth+1)
#     else:
#         print("Not cin/cout, depth=%d" % depth)
#         return False


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
    # if isinstance(node, BinaryOp) and node.op == '>>':
    #     cppin = cppin_number(node, 0)
    #     if type(cppin) == int:
    #         directin += cppin
    #         print(cppin)
    # TODO: 添加cout数量统计
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


def feature_extract(filename):
    """提取给定文件的特征向量"""
    ast = parse_file(filename, use_cpp=False)
    directin, loopin, condin = extract_stdin(ast)
    return directin,loopin,condin


if __name__ == '__main__':
    print(feature_extract('test.txt'))
