import numpy as np
import operator
import os
import time

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']

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



def showdatas(datingDataMat, datingLabels):
    """
    函数说明：可视化数据

    Parameters:
        datingDataMat - 特征矩阵
        datingLabels - 分类Label
        
    Returns:
        None
        
    Modify:
        2020-05-23
    """
    # 将fig画布分隔成1行1列，不共享x轴和y轴，fig画布的大小为（13，8）
    # 当nrows=2，ncols=2时，代表fig画布被分为4个区域，axs[0][0]代表第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))
    # 获取datingLabels的行数作为label的个数
    # numberOfLabels = len(datingLabels)
    # label的颜色配置矩阵
    LabelsColors = []
    for i in datingLabels:
        # didntLike
        if i == 1:
            LabelsColors.append('black')
        # smallDoses
        if i == 2:
            LabelsColors.append('orange')
        # largeDoses
        if i == 3:
            LabelsColors.append('red')
    # 画出散点图，以datingDataMat矩阵第一列为x，第二列为y，散点大小为15, 透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors, s=15, alpha=.5)
    # 设置标题，x轴label， y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比')
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')
    # 画出散点图，以datingDataMat矩阵第一列为x，第三列为y，散点大小为15, 透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题，x轴label， y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰淇淋公升数')
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数')
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰淇淋公升数')
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')
    # 画出散点图，以datingDataMat矩阵第二列为x，第三列为y，散点大小为15, 透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题，x轴label， y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰淇淋公升数')
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比')
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰淇淋公升数')
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()


# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet, ranges, minVals


# 分类器针对约会网站的测试代码
def datingClassTest():
    """
    函数说明：对已知数据进行分类。
    读入数据，归一化，取10%数据作为测试机，其他数据作为训练集

    Parameters:
        filename - 文件名
        
    Returns:
        returnMat - 特征矩阵
        classLabelVector - 分类label向量
        
    Modify:
        2020-05-18
    """
    # 取所有数据的10% hoRatio越小，错误率越低
    hoRatio = 0.10

    # 将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    filename = "datingTestSet.txt"
    current_path = os.path.dirname(__file__)
    datingDataMat, datingLabels = file2matrix(current_path+"/"+filename)
    # 数据归一化，返回归一化数据结果，数据范围，最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获取normMat的行数
    m = normMat.shape[0]
    # 10%的测试数据的个数
    numTestVecs = int(m * hoRatio)
    # 分类错误计数
    errorCount = 0.0
    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集，后m-numTestVecs个数据作为训练集
        # k选择label数+1（结果比较好）
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],\
                                     datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" % (errorCount/float(numTestVecs)*100))


def classifyPerson():
    """
    函数说明：通过输入一个人的三围特征，进行分类输出

    Parameters:
        None
        
    Returns:
        None
        
    Modify:
        2020-05-23
    """
    # 输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    # 三维特征用户输入
    percentTats = float(input("玩视频游戏所消耗时间百分比："))
    ffMiles = float(input("每年获得的飞行常客里程数："))
    iceCream = float(input("每周消费的冰淇淋公升数："))
    # 打开的文件名
    filename = "datingTestSet.txt"
    # 打开并处理数据
    current_path = os.path.dirname(__file__)
    datingDataMat, datingLabels = file2matrix(current_path+"/"+filename)
    # 训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 生成NumPy数组，测试集
    inArr = np.array([percentTats, ffMiles, iceCream])
    # 测试集归一化
    norminArr = (inArr - minVals) / ranges
    # 返回分类结果
    classifierResult = classify0(norminArr, normMat, datingLabels, 4)
    # 打印结果
    print("你可能%s这个人" % (resultList[classifierResult - 1]))


def main():
    start = time.time()
    current_path = os.path.dirname(__file__)
    # 打开文件的名称
    filename = "datingTestSet.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2matrix(current_path + '/' + filename)
    # print(datingDataMat)
    # print(datingLabels)

    # # 数据可视化
    # showdatas(datingDataMat, datingLabels)
    # 训练集归一化
    normDataset, ranges, minVals = autoNorm(datingDataMat)
    # print(normDataset)
    # print(ranges)
    # print(minVals)
    # classifyPerson()

    # 创建数据集
    group, labels = createDataSet()
    # 测试集
    test = [100,100]
    # kNN分类
    test_class = classify0(test, group, labels, 3)
    # 打印分类结果
    print(test_class)
    end = time.time()
    # 打印程序运行时间
    print('Running time: %f Seconds'%(end-start))


if __name__ == "__main__":
    # filename = "datingTestSet.txt"
    # current_path = os.path.dirname(__file__)
    # datingDataMat, datingLabels = file2matrix(current_path+"/"+filename)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    # plt.show()

    main()