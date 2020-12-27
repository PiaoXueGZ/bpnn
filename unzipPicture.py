from PIL import Image


imagefile = open("D:\\lgs\\MySoftware\\python\\bpNN\\trainingdata\\train-images.idx3-ubyte", "rb")
imagefile.seek(16)
lablefile = open("D:\\lgs\\MySoftware\\python\\bpNN\\trainingdata\\train-labels.idx1-ubyte", "rb")
lablefile.seek(8)
for i in range(100):
    im = Image.frombytes("L", (28, 28), imagefile.read(784))
    lable = lablefile.read(1)
    im.save("D:\\lgs\\MySoftware\\python\\bpNN\\image\\" + str(i) + "_" + str(lable[0]) + ".png", "PNG")

