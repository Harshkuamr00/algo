import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
x=2*np.random.rand(100,1)# 2d array where (100,1) , 100 samples , 1 independent variable where range is (0,2)
y=4+3*x+np.random.rand(100,1) # y is dependent variable as u can see 

# create and fit the linear regression model
model=LinearRegression()
model.fit(x,y)

# make prediction
y_pred=model.predict(x)

# here we can check our values
df=pd.DataFrame(data=np.column_stack((x,y)),columns=['Independent Variable (x)','Independent Variable (y)'])
df.to_csv("values of independent variable and dependent variable (x,y).csv")
#print(df.info())

# plotting the data and the regression line
plt.scatter(x,y,color='blue',label='Data points')
plt.plot(x,y,color='red',label='Regression line')
plt.plot(y_pred,color='green',label='Prediction line estimate')
plt.title('Linear Regression')
plt.xlabel('Independent Variable (x)')
plt.ylabel('Dependent Variable (y)')
plt.legend()
plt.show()


# print coefficients and intercept
# we use it because it has some curical role like define the relationship between the Independent variable and dependent variable
# coefficient : or (slope) indicates how much the dependent variable (y) is expected to change for one -unit of increasing independent variable x
# intercept : is the expected the values of the dependent variable when all independent variable are equal to zero
# regression model is: y=intercept+(coefficient×X)

coefficients=model.coef_
intercept=model.intercept_

print(f'''
Coefficient : {coefficients}
Intercept : {intercept}''')