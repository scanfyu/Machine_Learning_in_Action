import numpy as np
import operator

def createDataSet():
    # 四组二维特征
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    # 四组特征的标签
    labels = ['A','A','B','B']
    return group, labels



# 2-1函数：kNN算法
def classify0(inX, dataSet, labels, k):
    """
    函数说明：kNN算法，分类器

    Parameters:
        inX - 用于分类的数据（测试集）
        dataSet - 用于训练的数据（训练集）（n*1维列向量）
        labels - 分类标准（n*1维列向量）
        k - kNN算法参数，选择距离最小的k个点
        
    Returns:
        sortedClassCount[0][0] - 分类结果
        
    Modify:
        2020-05-17
    """
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 将inX重复dataSetSize次并排成一列
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方（用diffMat的转置乘diffMat）
    sqDiffMat = diffMat**2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方，计算出距离
    distances = sqDistances**0.5
    # argsort函数返回的是distances值从小到大的--索引值
    sortedDistIndicies = distances.argsort()
    # 定义一个记录类别次数的字典
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        # 字典的get()方法，返回指定键的值，如果值不在字典中返回0
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # python3中用items()替换python2中的iteritems()
    # key = operator.itemgetter(1)根据字典的值进行排序
    # key = operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1), reverse = True)
    # 返回次数最多的类别，即所要分类的类别
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    函数说明：打开解析文件，对数据进行分类，1代表不喜欢，2代表魅力一般，3代表极具魅力

    Parameters:
        filename - 文件名
        
    Returns:
        returnMat - 特征矩阵
        classLabelVector - 分类label向量
        
    Modify:
        2020-05-18
    """
    # 打开文件
    fr = open(filename)
    # 读取文件所有内容
    arrayOlines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOlines)
    # 返回的NumPy矩阵numberOfLines行，3列
    returnMat = np.zeros((numberOfLines, 3))
    # 创建分类标签向量
    classLabelVector = []
    # 行的索引值
    index = 0
    # 读取每一行
    for line in arrayOlines:
        # 去掉每一行首尾的空白符，例如'\n','\r','\t',' '
        line = line.strip()
        # 将每一行内容根据'\t'符进行切片,本例中一共有4列
        listFromLine = line.split('\t')
        # 将数据的前3列进行提取保存在returnMat矩阵中，也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        # 根据文本内容进行分类1：不喜欢；2：一般；3：喜欢
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    # 返回标签列向量以及特征矩阵
    return returnMat, classLabelVector


if __name__ == "__main__":
    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    print(type(datingDataMat))
    print(type(datingLabels))