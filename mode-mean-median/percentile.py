import numpy as np
def cal_percentile(data,percentile):
    return np.percentile(data,percentile)
    
data1=np.arange(1,11)
data2=np.arange(11,21)
print(f'''
Dataset 1 : {data1}
Dataset 2 : {data2}''')

print("Percentile of data1 75 percentile : ",cal_percentile(data1,percentile=75))
print("Percentile of data2 50 percentile : ",cal_percentile(data2,percentile=50))