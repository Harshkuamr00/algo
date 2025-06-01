from statistics import mode
import numpy as np
def mean(data):
    n=len(data)
    s=0
    for i in range(n):
        s+=i
    return s//n
def median(data):
    data.sort()
    n=len(data)
    mid=n//2
    return (data[mid-1]+data[mid]) //2 if n%2==0 else data[mid]

def mo(data):
    return mode(data)


data=np.random.random(8)*10
data=data.astype(int)
print("Your dataset : ",data)
print("Mean of the data : ",mean(data))
print("Mode of the data : ",mo(data))
print("Median of the data : ",median(data))

