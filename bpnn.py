import numpy as np
import random

trainDataPath = "D:\\lgs\\MySoftware\\python\\bpNN\\trainingdata\\train-images.idx3-ubyte"
trainDataOffset = 16
trainLablePath = "D:\\lgs\\MySoftware\\python\\bpNN\\trainingdata\\train-labels.idx1-ubyte"
trainLableOffset = 8
testDataPath = "D:\\lgs\\MySoftware\\python\\bpNN\\trainingdata\\t10k-images.idx3-ubyte"
testDataOffset = 16
testLablePath = "D:\\lgs\\MySoftware\\python\\bpNN\\trainingdata\\t10k-labels.idx1-ubyte"
testLableOffset = 8

step = 0.01  #学习率

#读取数据
imageFile = open(trainDataPath, "rb")
imageFile.seek(trainDataOffset)
lableFile = open(trainLablePath, "rb")
lableFile.seek(trainLableOffset)
Datas = [[np.frombuffer(imageFile.read(784), np.uint8).reshape((1, 784)) / 255, lableFile.read(1)[0]] for x in range(60000)]

testImageFile = open(testDataPath, "rb")
testImageFile.seek(testDataOffset)
testLableFile = open(testLablePath, "rb")
testLableFile.seek(testLableOffset)
testDatas = [[np.frombuffer(testImageFile.read(784), np.uint8).reshape((1, 784)) / 255, testLableFile.read(1)[0]] for x in range(10000)]

k1, k2 = np.random.randint(0, 100, (785, 32)) / 100, np.random.randint(0, 100, (33, 10)) / 100  #初始化参数列表

def sigmoid(m):
    return 1 / (1 + np.exp(-m))

def dSigmoid(m):    #也等于sigmoid(m)*(1-sigmoid(m))
    expNegm = np.exp(-m)
    return expNegm / np.square((1 + expNegm))

def wantedArray(n):
    return np.array([1 if x == n else 0 for x in range(10)]).reshape((1, 10))

def training(dataList):
    global k1
    global k2
    dk1, dk2 = np.zeros((785, 32)), np.zeros((33, 10))

    for data in dataList:
        L1 = np.append(data[0], [1]).reshape((1, 785))  #加入偏置
        L2 = np.append(sigmoid(L1 @ k1), [1]).reshape((1, 33))  #加入偏置
        L3 = sigmoid(L2 @ k2)
        w = wantedArray(data[1])
        delta3 = (w - L3) * dSigmoid(L3)
        dk2 += L2.T @ delta3
        delta2 = (k2 @ delta3.reshape(10)) * dSigmoid(L2)
        dk1 += L1.T @ (delta2.reshape(33)[:-1].reshape((1, 32)))    #把delta2削掉最后一个数来适配大小
    
    dk1 /= len(dataList)
    dk2 /= len(dataList)
    #k1 -= step * dk1
    #k2 -= step * dk2
    k1 += step * dk1
    k2 += step * dk2


def loss(out, n):
    return 0.5 * np.sum(np.square(out - wantedArray(n)))

def getAns(m):
    L1 = np.append(m, [1]).reshape((1, 785))  #加入偏置
    L2 = np.append(sigmoid(L1 @ k1), [1]).reshape((1, 33))  #加入偏置
    L3 = sigmoid(L2 @ k2)
    ans = 0
    _max = L3[0][0]
    for i in range(1, 10):
        if L3[0][i] > _max:
            _max = L3[0][i]
            ans = i
    return ans



for i in range(1):
    random.shuffle(Datas)
    for j in range(1200):
        training(Datas[j * 50 : (j + 1) * 50])

counter = 0
for data in testDatas:
    if getAns(data[0]) == data[1]:
        counter += 1

print(counter / 10000.0)