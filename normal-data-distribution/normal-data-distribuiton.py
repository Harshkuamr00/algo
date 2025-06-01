import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=np.random.normal(loc=40,scale=10,size=2000)

plt.hist(data,bins=30,edgecolor='grey',density=True,color='green')

df=pd.DataFrame(data)
df.to_csv('normal_distribution.csv')

plt.title("Normal_distribution - Generated randomly")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()
