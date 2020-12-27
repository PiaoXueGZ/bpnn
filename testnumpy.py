import numpy as np
from PIL import Image

'''
matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix1 @ matrix1)
'''

trainDataPath = "D:\\lgs\\MySoftware\\python\\bpNN\\trainingdata\\train-images.idx3-ubyte"
trainDataOffset = 16
trainLablePath = "D:\\lgs\\MySoftware\\python\\bpNN\\trainingdata\\train-labels.idx1-ubyte"
trainLableOffset = 8

imageFile = open(trainDataPath, "rb")
imageFile.seek(trainDataOffset)
lableFile = open(trainLablePath, "rb")
lableFile.seek(trainLableOffset)

#Datas = [[np.frombuffer(imageFile.read(784), np.uint8).reshape((1, 784)) / 255, lableFile.read(1)[0]] for x in range(100)]
Datas = [[np.frombuffer(imageFile.read(784), np.uint8).reshape((1, 784)), lableFile.read(1)[0]] for x in range(100)]
im = Image.fromarray(Datas[0][0].reshape((28, 28)), "L")
im.save("D:\\lgs\\MySoftware\\python\\bpNN\\image\\test.png", "PNG")