from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

x1=np.random.rand(100,1)
x2=np.random.rand(100,1) *10
x=np.hstack((x1,x2))

y=2*x1+2*x2 +np.random.rand(100,1)

# check the values 
df=pd.DataFrame(data=np.column_stack((x,y)),columns=['Random Variable (x1)','Random Variable (x2)','Dependent Variable (y)'])
#print(df.head())
#print(df.isnull().sum())

df.to_csv('Variable.csv')

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print(f'''Training set size : {len(x_train)}
Testing set size : {len(x_test)}''')