import numpy as np
from PIL import Image

testDataPath = "D:\\lgs\\MySoftware\\python\\bpNN\\trainingdata\\t10k-images.idx3-ubyte"
testDataOffset = 16
testLablePath = "D:\\lgs\\MySoftware\\python\\bpNN\\trainingdata\\t10k-labels.idx1-ubyte"
testLableOffset = 8

testImageFile = open(testDataPath, "rb")
testImageFile.seek(testDataOffset)
testLableFile = open(testLablePath, "rb")
testLableFile.seek(testLableOffset)
testDatas = [[np.frombuffer(testImageFile.read(784), np.uint8).reshape((1, 784)) / 255, testLableFile.read(1)[0]] for x in range(10000)]


z = np.load("model.npz")
k1, k2, b1, b2 = z["arr_0"], z["arr_1"], z["arr_2"], z["arr_3"]

def sigmoid(m):
    return 1 / (1 + np.exp(-m))

def dSigmoid(m):    #也等于sigmoid(m)*(1-sigmoid(m))
    expNegm = np.exp(-m)
    return expNegm / np.square((1 + expNegm))

def wantedArray(n):
    return np.array([1 if x == n else 0 for x in range(10)]).reshape((1, 10))

def loss(out:np.ndarray, n:int) -> float:
    return 0.5 * np.sum(np.square(out - wantedArray(n)))

def getAns(m:np.ndarray) -> int:
    L1 = m
    L2 = sigmoid(L1 @ k1 + b1)
    L3 = sigmoid(L2 @ k2 + b2)
    return np.argmax(L3)

'''
path = input('Input the path of image:')
im = Image.open(path).convert('L')
in_data = (1 - np.asarray(im, dtype=np.uint8) / 255).reshape((1, 784))

L2 = sigmoid(in_data @ k1 + b1)
L3 = sigmoid(L2 @ k2 + b2)
L3 = L3.reshape((10,))
for i in range(10):
    print(i, "  possibility  ", L3[i])
print("Biggest possibility:", np.argmax(L3))
i = input('Input ans:')
print("Loss=", loss(L3, i))
'''

counter = 0
sumOfLoss = 0
for pair in testDatas:
    L3 = sigmoid(sigmoid(pair[0] @ k1 + b1) @ k2 + b2).reshape((10,))
    sumOfLoss += loss(L3, pair[1])
    if np.argmax(L3) == pair[1]:
        counter += 1

print(counter)
print(sumOfLoss / 10000)