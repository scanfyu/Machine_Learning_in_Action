from math import log

def calcShannonEnt(dataSet):
    """
    函数说明：计算给定数据集的经验熵（香农熵）
            Ent（D） = -SUM（kp*Log2（kp））

    Parameters:
        dataSet - 数据集
        
    Returns:
        shannonEnt - 经验熵（香农熵）

    Modify:
        2020-05-24
    """
    # 返回数据集的行数
    numEntires = len(dataSet)
    # 保存每个标签（Label）出现次数的“字典”
    labelCounts = {}
    # 对每组特征向量进行统计
    for featVec in dataSet:
        # 提取标签（Label）信息
        currentLabel = featVec[-1]
        # 如果标签（Label）没有放入统计次数的字典，添加进去
        if currentLabel not in labelCounts.keys():
            # 创建一个新的键值对，键为currentLabel值为0
            labelCounts[currentLabel] = 0
        # Label计数
        labelCounts[currentLabel] += 1
    # 经验熵（香农熵）
    shannonEnt = 0.0
    # 计算香农熵
    for key in labelCounts:
        # 选择该标签（Label）的概率
        prob = float(labelCounts[key]) / numEntires
        # 利用公式计算
        shannonEnt -= prob*log(prob, 2)
    # 返回经验熵（香农熵）
    return shannonEnt


def createDataSet():
    """
    函数说明：创建测试数据集

    Parameters:
        None
        
    Returns:
        dataSet - 数据集
        labels - 分类属性

    Modify:
        2020-05-24
    """
    # 数据集
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    # 分类属性
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    # 返回数据集和分类属性
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    函数说明：按照给定特征划分数据集

    Parameters:
        dataSet - 待划分的数据集
        axis - 划分数据集的特征
        values - 需要返回的特征的值
        
    Returns:
        None
        
    Modify:
        2020-05-24
    """
    # 创建返回的数据集列表
    retDataSet = []
    # 遍历数据集的每一行
    for featVec in dataSet:
        if featVec[axis] == value:
            # 去掉axis特征
            reducedFeatVec = featVec[:axis]
            # 将符合条件的添加到返回的数据集
            # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            reducedFeatVec.extend(featVec[axis+1:])
            # 列表中嵌套列表
            retDataSet.append(reducedFeatVec)
    # 返回划分后的数据集
    return retDataSet


if __name__ == "__main__":
    pass