import kNN

group, labels = kNN.createDataSet()

# print(group)
# print(labels)


# 2-1函数测试
output_1 = kNN.classify0([0,0], group, labels, 3)
print(output_1)