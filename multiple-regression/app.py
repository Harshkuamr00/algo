import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
x1=np.random.rand(100,1)*100
x2=np.random.rand(100,1)*200
x1 = x1.astype(int)
x2 = x2.astype(int)
x=np.hstack((x1,x2))# 2d array of two independent variable

y= x1 - x2 +np.random.randn(100,1) # dependent variable with noise


# check the values of our variables
df=pd.DataFrame(data=np.column_stack((x,y)),columns=['Independent Variable (x1)','Independent Variable (x2)','Dependent Variable (y)'])
print(df.info())
df.to_csv("variables.csv",index=False)


# model create and fit
model=LinearRegression()
model.fit(x,y)

# make prediction
y_pred=model.predict(x)

#plot the variable and other stuff
fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111,projection='3d')

# scatter plot of the original data
#ax.scatter(x1,x2,y,color='blue',label='Data point') for only single color
scatter=ax.scatter(x1,x2,y,c=y.flatten(),cmap='viridis',s=50,alpha=0.7,label='Data points')

# add color bar this line from ai 
plt.colorbar(scatter,ax=ax,shrink=0.5,aspect=20,label='Dependent Variable (y) Values')

# grid for the regression plane , just we are make plane to project like differance is same like number line of the plot
x1_range=np.linspace(x1.min(),x1.max(),10)
x2_range=np.linspace(x2.min(),x2.max(),10)
x1_grid,x2_grid=np.meshgrid(x1_range,x2_range)

# calculate the corresponding y values for the regression plane
#y_grid=model.intercept_ + model.coef_[0] * x1_grid +model.coef_[1] * x2_grid
coef = model.coef_.reshape(-1)  # Ensure 1D array
y_grid = model.intercept_ + coef[0] * x1_grid + coef[1] * x2_grid

# plot a plane fot it
ax.plot_surface(x1_grid,x2_grid,y_grid,color='red',alpha=0.3)

# adding labels and tittles
ax.set_xlabel('Independent Variable (x1)')
ax.set_ylabel('Independent Variable (x2)')
ax.set_zlabel('Dependent Variable (y)')
ax.set_title('Multiple Linear Regression Visualization')

plt.legend()
plt.show()