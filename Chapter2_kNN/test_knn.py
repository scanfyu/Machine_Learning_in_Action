import kNN
import os

# group, labels = kNN.createDataSet()

# print(group)
# print(labels)


# 2-1函数测试
# output_1 = kNN.classify0([0,0], group, labels, 3)
# print(output_1)


# 2-2文本记录到转换Numpy
# current_path = os.path.dirname(__file__)
# print(current_path)
# filename = "datingTestSet.txt"
# datingDataMat, datingLabels = kNN.file2matrix(current_path+"/"+filename)

# normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
# print(normMat)
# print(ranges)
# print(minVals)


# 2-4分类器针对约会网站的测试代码
kNN.datingClassTest()