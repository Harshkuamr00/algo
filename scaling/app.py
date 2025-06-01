import numpy as np
from sklearn.preprocessing import MinMaxScaler # Normalization
from sklearn.preprocessing import StandardScaler # standardization 
import pandas as pd

'''Transformation and scaling are essential preprocessing steps in machine learning (ML) 
that prepare data for better model performance and reliability.
'''
np.random.seed(42)

data = np.random.rand(100,1) *100 +200 # create the random data 

# create and fit scaling model approach Normalization regression is used that
scalar = MinMaxScaler()
# fit the data transform  means it convert feature into 0 and 1 so model can train and test the data
scaled_data=scalar.fit_transform(data)

df=pd.DataFrame(data,columns=['Random vairable'])
df.to_csv("variable.csv",index=False)

print(f'''
Original Data : 
{data[:10]}
Scaled Data : 
{scaled_data[:10]}\n''')

# another approach Standardization 

data1=np.random.normal(loc=40,scale=10,size=1000)

scale=StandardScaler()
Scaledx=scale.fit_transform(data)

cf=pd.DataFrame(data1,columns=['used for standardization variable'])
cf.to_csv('another-variable.csv',index=False)

print(f'''
Original Data :
{data1[:10]}
Scaled Data By Standardization :
{Scaledx[:10]}''')


