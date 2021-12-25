import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv('potato.csv')
plt.scatter(df['Date'],df['p/kg'])
plt.show()