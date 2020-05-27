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


def createVocabList(dataSet):
    """
    函数说明：将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

    Parameters:
        dataSet - 整理的样本数据集

    Returns:
        vocabSet - 返回不重复的词条列表，也就是词汇表

    Modify:
        2020-05-27
    """
    # 创建一个空的不重复列表
    # set是一个无序且不重复的元素集合
    vocabSet = set([])
    for document in dataSet:
        # 取并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def trainNB0(trainMatrix, trainCategory):
    """
    函数说明：朴素贝叶斯分类器训练函数

    Parameters:
        trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
        trainCategory - 训练类标签向量，即loadDataSet返回的classVec

    Returns:
        p0Vect - 侮辱类的条件概率数组
        p1Vect - 非侮辱类的条件概率数组
        pAbusive - 文档属于侮辱类的概率

    Modify:
        2020-05-27
    """
    # 计算训练文档数目
    numTrainDocs = len(trainMatrix)
    # 计算每篇文档的词条数目
    numWords = len(trainMatrix[0])
    # 文档属于侮辱类的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 创建numpy.zeros数组，词条出现数初始化为0
    # p0Num = np.zeros(numWords)
    # p1Num = np.zeros(numWords)
    # 创建numpy.ones数组，词条出现数初始化为1,拉普拉斯平滑
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    # 分母初始化为0
    # p0Denom = 0.0
    # p1Denom = 0.0
    # 分母初始化为2，拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)...
        if trainCategory[i] == 1:
            # 统计所有侮辱类文档中每个单词出现的个数
            p1Num += trainMatrix[i]
            # 统计一共出现的侮辱单词的个数
            p1Denom += sum(trainMatrix[i])
        # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)...
        else:
            # 统计所有非侮辱类文档中每个单词出现的个数
            p0Num += trainMatrix[i]
            # 统计一共出现的非侮辱单词的个数
            p0Denom += sum(trainMatrix[i])
    # 每个侮辱类单词分别出现的概率
    # p1Vect = p1Num / p1Denom
    # 取对数，防止下溢出
    p1Vect = np.log(p1Num / p1Denom)
    # 每个非侮辱类单词分别出现的概率
    # p0Vect = p0Num / p0Denom
    # 取对数，防止下溢出
    p0Vect = np.log(p0Num / p0Denom)
    # 返回属于侮辱类的条件概率数组、属于非侮辱类的条件概率数组、文档属于侮辱类的概率
    return p0Vect, p1Vect, pAbusive