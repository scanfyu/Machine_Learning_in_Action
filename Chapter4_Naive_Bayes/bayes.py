#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Machine_Learning_in_Action -> bayes.py
@Python ：Python 3.5+
@Author ：Sunny
@Date   ：2020/5/25/0025 19:27
@Desc   ：
=================================================='''

import numpy as np
from functools import reduce


def loadDataSet():
    """
    函数说明：创建实验样本

    Parameters:
        None

    Returns:
        postingList - 实验样本切分的词条
        classVec - 类别标签向量

    Modify:
        2018-07-21
    """
    # 切分的词条
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量，1代表侮辱性词汇，0代表不是
    classVec = [0, 1, 0, 1, 0, 1]
    # 返回实验样本切分的词条、类别标签向量
    return postingList, classVec


def createVocabList(dataSet):
    """
    函数说明：将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

    Parameters:
        dataSet - 整理的样本数据集

    Returns:
        vocabSet - 返回不重复的词条列表，也就是词汇表

    Modify:
        2018-07-21
    """
    # 创建一个空的不重复列表
    # set是一个无序且不重复的元素集合
    vocabSet = set([])
    for document in dataSet:
        # 取并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    函数说明：根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

    Parameters:
        vocabList - createVocabList返回的列表
        inputSet - 切分的词条列表

    Returns:
        returnVec - 文档向量，词集模型

    Modify:
        2018-07-21
    """
    # 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    # 遍历每个词条
    for word in inputSet:
        if word in vocabList:
            # 如果词条存在于词汇表中，则置1
            # index返回word出现在vocabList中的索引
            # 若这里改为+=则就是基于词袋的模型，遇到一个单词会增加单词向量中德对应值
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary" % word)
    # 返回文档向量
    return returnVec