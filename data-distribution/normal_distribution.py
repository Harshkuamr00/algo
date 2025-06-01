
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import joblib

#def cal_normal_distribution(data):
    
data=np.random.normal(loc=50,scale=10,size=1000).astype(int) # random variable where loc=50 stands for mean is 50 this is the centre
# scale is stands for standard deviation separate by 10
# size is stands for number of variable in array
df=pd.DataFrame(data)
df.to_csv("Mice weight.csv")

# by matplotlib we can visulize the data distribution in chart or graph form 
plt.hist(data,bins=10,edgecolor='black')
plt.title("Histogram - Data Distribution")# title part 
plt.xlabel("mice Weight")# x-axis
plt.ylabel("No of Mice")# y-axis
plt.show()
