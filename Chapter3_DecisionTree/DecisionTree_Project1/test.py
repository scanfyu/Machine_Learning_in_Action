import trees

myDat, labels = trees.createDataSet()
print(myDat)

# re_data = trees.calcShannonEnt(myDat)
# print(re_data)

# set_1 = trees.splitDataSet(myDat, 0, 1)
# set_2 = trees.splitDataSet(myDat, 0, 0)
# print(set_1)
# print(set_2)

best_f = trees.createTree(myDat, labels)
print(best_f)