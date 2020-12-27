import math

step = 0.1
r = 2

def F(x):
    return math.exp(x) + math.exp(-x)

def dF(x):
    return math.exp(x) - math.exp(-x)

def iterFunc1(x):
    return x / (1 + x * x)

def iterFunc2(x):
    if(x > 1):
        return 1
    elif(-1 < x < 1):
        return x
    else:
        return -1
    
def iterFunc3(x):
    return step * x

x = 2
counter = 0
while math.fabs(x) >= 0.0001:
    print(x)
    x -= iterFunc3(dF(x))
    counter += 1
    if counter % 100000 == 0:
        print(counter, x, sep=" ")

print(counter)