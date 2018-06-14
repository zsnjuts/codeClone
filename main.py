# -*- coding:utf-8 -*-
# ------------------------------------------------------------------------------
# pycparser: c-to-c.py
#
# Example of using pycparser.c_generator, serving as a simplistic translator
# from C to AST and back to C.
#
# Eli Bendersky [http://eli.thegreenplace.net]
# License: BSD
# ------------------------------------------------------------------------------
from pycparser import parse_file, c_generator
from pycparser.c_ast import *


def translate_to_c(filename):
    """ Simply use the c_generator module to emit a parsed AST.
    """
    ast = parse_file(filename, use_cpp=False)
    ast.show(attrnames=True, nodenames=True)
    # print(visit(ast))


def visit(node):
    """vist node"""
    cnt = 0
    if isinstance(node, For):
        cnt += 1
    for child in node.children():
        cnt += visit(child[1])
    return cnt


if __name__ == '__main__':
    translate_to_c('dataset/train/a98e/0bf3f6043f564c86.txt')
