import numpy as np

'''
matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix1 @ matrix1)
'''

f = open("D:\\lgs\\MySoftware\\python\\bpNN\\trainingdata\\t10k-images.idx3-ubyte", "rb")
f.seek(16)

#vector1 = np.array(f.read(28*28), np.uint8)
#print(vector1)